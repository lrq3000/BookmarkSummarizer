# Copyright 2024 wyj
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import requests
from bs4 import BeautifulSoup
import json
import os
import re
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import chardet
from tqdm import tqdm
import traceback
from browser_history.browsers import *
import hashlib
import threading

# TOML parsing imports with fallback for older Python versions
try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Fallback for older versions


# --- Browser Profile Configuration ---
# Default profile paths are handled by browser_history module
# Users can specify custom paths via --profile-path argument
# ----------------------------------------------------

bookmarks_path = os.path.expanduser("./bookmarks.json")
bookmarks_with_content_path = os.path.expanduser("./bookmarks_with_content.json")
failed_urls_path = os.path.expanduser("./failed_urls.json")

# Global sets for deduplication
url_hashes = set()
content_hashes = set()
content_lock = threading.Lock()

# Load TOML configuration
def load_config(config_path="default_config.toml"):
    """
    Load configuration from TOML file.

    Parameters:
        config_path (str): Path to the TOML configuration file.

    Returns:
        dict: Configuration dictionary loaded from TOML file.
    """
    try:
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        return config
    except FileNotFoundError:
        print(f"Warning: Configuration file '{config_path}' not found. Using default values.")
        return {}
    except Exception as e:
        print(f"Warning: Error loading configuration from '{config_path}': {e}. Using default values.")
        return {}

# Configuration settings
class ModelConfig:
    # Supported model types
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"
    OLLAMA = "ollama"  # Added Ollama model type

    def __init__(self, config_data=None):
        """
        Initialize ModelConfig with TOML configuration data.

        Parameters:
            config_data (dict, optional): Configuration dictionary from TOML file.
                                         If None, uses default values.
        """
        if config_data is None:
            config_data = {}

        # Extract model section from config
        model_config = config_data.get("model", {})

        # Default configuration with TOML overrides
        self.model_type = model_config.get("model_type", self.OPENAI)
        self.api_key = model_config.get("api_key", "")
        self.api_base = model_config.get("api_base", "https://api.openai.com/v1")
        self.model_name = model_config.get("model_name", "gpt-3.5-turbo")
        self.max_tokens = model_config.get("max_tokens", 1000)
        self.temperature = model_config.get("temperature", 0.3)

        # Extract crawl section from config
        crawl_config = config_data.get("crawl", {})
        self.max_input_content_length = crawl_config.get("max_input_content_length", 6000)
        self.generate_summary = crawl_config.get("generate_summary", True)

        # DeepSeek specific configuration (keeping defaults for backward compatibility)
        self.top_p = model_config.get("top_p", 0.7)
        self.top_k = model_config.get("top_k", 50)
        self.frequency_penalty = model_config.get("frequency_penalty", 0.5)
        self.system_prompt = model_config.get("system_prompt", "")
        self.use_tools = model_config.get("use_tools", False)

        # Qwen specific configuration
        self.qwen_api_version = model_config.get("qwen_api_version", "2023-12-01-preview")
        # Ollama specific configuration
        self.ollama_format = model_config.get("ollama_format", "text")  # Options: json, text

# Generate summary using a Large Language Model (LLM)
def generate_summary(title, content, url, config=None):
    """
    Generates a summary of the webpage content using an LLM.
    
    Parameters:
        title (str): Webpage title
        content (str): Webpage content
        url (str): Webpage URL
        config (ModelConfig, optional): Model configuration, defaults to environment variables.
        
    Returns:
        str: The generated summary.
    """
    if config is None:
        config = ModelConfig()
    
    try:
        # Limit content length to avoid exceeding token limits
        max_content_length = config.max_input_content_length
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        
        # Construct a more detailed prompt
        prompt = f"""Generate only a comprehensive, informative summary (approx. 500 words) for the following webpage content. Begin directly with the summary text. Do not include any introductory phrases such as “Here is a summary” or similar.

Webpage Title: {title}
Webpage URL: {url}

Webpage Content:
{content}

Summary Requirements:
1. Automatically detect the language of the content and generate the summary in the same language.
2. Organize the content in a key-information-dense manner, ensuring the inclusion of important technical terms, entity names, and key concepts.
3. Use a clear paragraph structure, dividing information by topic, with each paragraph focusing on a core point.
4. Provide a concise introductory summary sentence at the beginning, briefly stating the main content and purpose of the document.
5. Use factual, specific statements, avoiding vague or general descriptions.
6. Retain important numbers, dates, names, technical terms, and unique identifiers from the original text.
7. For technical content, include specific technology names, version numbers, parameters, and method names.
8. For news events, clearly include the time, location, people, and key details of the event.
9. For tutorials or guides, list specific step names and critical operational points.
10. For products or services, include specific product names, features, and specifications.
11. Ensure high information density for easy vector retrieval matching.
12. Output only the summary text. Do not add any explanations, comments, or meta statements before or after the summary
13. Be concise, straight-to-the-point, and avoid unnecessary filler words, write in a note taking kind of way, without conjunctive words, only keep information dense words.

Please generate an information-dense, clearly structured summary, optimized for a text format suitable for vector retrieval, minimizing filler words, unnecessary repetition, and words like: 'okay', 'um', etc.
"""
        # Call the corresponding API based on the model type
        if config.model_type == ModelConfig.OLLAMA:
            return call_ollama_api(prompt, config)
        elif config.model_type == ModelConfig.QWEN:
            return call_qwen_api(prompt, config)
        elif config.model_type == ModelConfig.DEEPSEEK:
            return call_deepseek_api(prompt, config)
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")
    
    except Exception as e:
        print(f"Summary generation failed: {url} - {e}")
        return f"Summary generation failed: {str(e)}"

# Ollama API Call
def call_ollama_api(prompt, config=None):
    """
    API call specifically designed for models deployed with Ollama
    
    Parameters:
        prompt (str): The prompt text
        config (ModelConfig, optional): Model configuration
        
    Returns:
        str: The response text generated by the model
    """
    if config is None:
        config = ModelConfig()
    
    # Determine whether to use the chat or generate interface
    use_chat_api = True
    
    # API endpoint
    if use_chat_api:
        url = f"{config.api_base}/api/chat"
    else:
        url = f"{config.api_base}/api/generate"
    
    # Construct request payload
    if use_chat_api:
        # Use chat interface
        messages = [{"role": "user", "content": prompt}]
        
        # If there is a system prompt, add it to the messages
        if hasattr(config, 'system_prompt') and config.system_prompt:
            messages.insert(0, {"role": "system", "content": config.system_prompt})
        
        payload = {
            "model": config.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "num_predict": config.max_tokens
            }
        }
    else:
        # Use generate interface
        system_prompt = config.system_prompt if hasattr(config, 'system_prompt') and config.system_prompt else ""
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        payload = {
            "model": config.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "num_predict": config.max_tokens
            }
        }
    
    # Construct headers
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        # Send request
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=120  # Increase timeout, local models may require longer processing time
        )
        
        # Check response status
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        
        # Extract generated text - Ollama API format
        if use_chat_api:
            if "message" in result:
                return result["message"]["content"]
            elif "response" in result:
                return result["response"]
        else:
            if "response" in result:
                return result["response"]
        
        # If the expected field is not found, return the entire response
        return str(result)
            
    except requests.exceptions.RequestException as e:
        print(f"Ollama API Request Error: {e}")
        if 'response' in locals() and hasattr(response, 'text'):
            print(f"Response content: {response.text}")
        raise Exception(f"API call failed: {str(e)}")
    except ValueError as e:
        print(f"Ollama API Response Parsing Error: {e}")
        if 'response' in locals() and hasattr(response, 'text'):
            print(f"Response content: {response.text}")
        raise Exception(f"Response parsing failed: {str(e)}")
    except Exception as e:
        print(f"Ollama API Unknown Error: {e}")
        if 'response' in locals() and hasattr(response, 'text'):
            print(f"Response content: {response.text}")
        raise
      
# API call for Qwen (Tongyi Qianwen)
def call_qwen_api(prompt, config=None):
    """
    API call specifically designed for Qwen (Tongyi Qianwen) models.
    
    Parameters:
        prompt (str): The prompt text
        config (ModelConfig, optional): Model configuration
        
    Returns:
        str: The response text generated by the model
    """
    if config is None:
        config = ModelConfig()
    
    # API endpoint
    url = f"{config.api_base}/chat/completions"
    
    # Construct messages
    messages = [{"role": "user", "content": prompt}]
    
    # If there is a system prompt, add it to the messages
    if hasattr(config, 'system_prompt') and config.system_prompt:
        messages.insert(0, {"role": "system", "content": config.system_prompt})
    
    # Construct request payload - Qwen 2.5 is usually compatible with OpenAI format
    payload = {
        "model": config.model_name,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "top_p": config.top_p,
        "stream": False
    }
    
    # Construct headers
    headers = {
        "Content-Type": "application/json"
    }
    
    # If there is an API key, add it to the headers
    if config.api_key and config.api_key.strip():
        headers["Authorization"] = f"Bearer {config.api_key}"
    
    try:
        # Send request
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=60
        )
        
        # Check response status
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        
        # Extract generated text - Qwen API usually follows OpenAI format
        if "choices" in result and len(result["choices"]) > 0:
            if "message" in result["choices"]:
                return result["choices"]["message"]["content"]
            elif "text" in result["choices"]:
                return result["choices"]["text"]
            else:
                # If the expected field is not found, return the entire choice object
                return str(result["choices"])
        else:
            # If the response does not contain the choices field, return the entire response
            return str(result)
            
    except requests.exceptions.RequestException as e:
        print(f"Qwen API Request Error: {e}")
        if 'response' in locals() and hasattr(response, 'text'):
            print(f"Response content: {response.text}")
        raise Exception(f"API call failed: {str(e)}")
    except ValueError as e:
        print(f"Qwen API Response Parsing Error: {e}")
        if 'response' in locals() and hasattr(response, 'text'):
            print(f"Response content: {response.text}")
        raise Exception(f"Response parsing failed: {str(e)}")
    except Exception as e:
        print(f"Qwen API Unknown Error: {e}")
        if 'response' in locals() and hasattr(response, 'text'):
            print(f"Response content: {response.text}")
        raise

def call_deepseek_api(prompt, config=None):
    """
    API call specifically designed for DeepSeek R1.
    
    Parameters:
        prompt (str): The prompt text
        config (ModelConfig, optional): Model configuration
        
    Returns:
        str: The response text generated by the model
    """
    if config is None:
        config = ModelConfig()
    
    # API endpoint
    url = f"{config.api_base}/chat/completions"
    print(f"Calling DeepSeek API: {url}")
    print(f"Using model: {config.model_name}")
    print(f"API Key Length: {len(config.api_key) if config.api_key else 0}")
    
    # Construct messages
    messages = [{"role": "user", "content": prompt}]
    
    # If there is a system prompt, add it to the messages
    if hasattr(config, 'system_prompt') and config.system_prompt:
        messages.insert(0, {"role": "system", "content": config.system_prompt})
    
    # Construct request payload
    payload = {
        "model": config.model_name,  # e.g., "deepseek-ai/DeepSeek-R1"
        "messages": messages,
        "stream": False,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": getattr(config, 'top_p', 0.7),
        "top_k": getattr(config, 'top_k', 50),
        "frequency_penalty": getattr(config, 'frequency_penalty', 0.5),
        "n": 1,
        "response_format": {"type": "text"}
    }
    
    # Print request configuration for debugging
    print(f"Request config: temperature={config.temperature}, max_tokens={config.max_tokens}")
    
    # Construct headers
    headers = {
        "Content-Type": "application/json"
    }
    
    # If there is an API key, add it to the headers
    if config.api_key and config.api_key.strip():
        headers["Authorization"] = f"Bearer {config.api_key}"
        print("Authorization header added")
    else:
        print("API key not set, request does not include Authorization header")
    
    try:
        # Send request
        print("Sending request...")
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=60  # Increase timeout, as LLMs may require longer processing time
        )
        
        # Check response status
        print(f"Response status code: {response.status_code}")
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        print(f"Successfully received response: {result.keys() if isinstance(result, dict) else 'Non-dictionary response'}")
        
        # Extract generated text
        if "choices" in result and len(result["choices"]) > 0:
            if "message" in result["choices"]:
                content = result["choices"]["message"]["content"]
                print(f"Successfully extracted content, length: {len(content)}")
                return content
            elif "text" in result["choices"]:
                text = result["choices"]["text"]
                print(f"Successfully extracted text, length: {len(text)}")
                return text
            else:
                # If the expected field is not found, return the entire choice object
                print(f"Content or text field not found, returning entire choice object: {result['choices']}")
                return str(result["choices"])
        else:
            # If the response does not contain the choices field, return the entire response
            print(f"Response does not contain choices field, returning entire response: {result}")
            return str(result)
            
    except requests.exceptions.RequestException as e:
        print(f"DeepSeek API Request Error: {e}")
        if 'response' in locals() and hasattr(response, 'text'):
            print(f"Response content: {response.text}")
        raise Exception(f"API call failed: {str(e)}")
    except ValueError as e:
        print(f"DeepSeek API Response Parsing Error: {e}")
        if 'response' in locals() and hasattr(response, 'text'):
            print(f"Response content: {response.text}")
        raise Exception(f"Response parsing failed: {str(e)}")
    except Exception as e:
        print(f"DeepSeek API Unknown Error: {e}")
        if 'response' in locals() and hasattr(response, 'text'):
            print(f"Response content: {response.text}")
        raise

def test_api_connection(config=None):
    """Test if API connection is normal"""
    if config is None:
        config = ModelConfig()
    
    print(f"==== API Connection Test ====")
    print(f"Model Type: {config.model_type}")
    print(f"API Base URL: {config.api_base}")
    print(f"Model Name: {config.model_name}")
    print(f"API Key Length: {len(config.api_key) if config.api_key else 0}")
    
    try:
        # Simple test prompt
        test_prompt = "Who are you? Answer briefly."
        print(f"Test Prompt: '{test_prompt}'")
        
        print(f"Starting API connection test...")
        response = None
        
        # Call the corresponding API based on model type
        print(f'DEBUGLINE: config.model_type = {config.model_type}')
        if config.model_type == ModelConfig.OLLAMA:
            print(f"Using Ollama API")
            response = call_ollama_api(test_prompt, config)
        elif config.model_type == ModelConfig.QWEN:
            print(f"Using Qwen API")
            response = call_qwen_api(test_prompt, config)
        elif config.model_type == ModelConfig.DEEPSEEK:
            print(f"Using DeepSeek API")
            response = call_deepseek_api(test_prompt, config)
        else:
            # Handling for other model types...
            print(f"Unrecognized model type: {config.model_type}, attempting to use DeepSeek API")
            response = call_deepseek_api(test_prompt, config)
        
        # Check response
        if response and isinstance(response, str) and len(response) > 0:
            print("✅ API connection test successful!")
            print(f"Model Response: {response[:100]}...")
            return True
        else:
            print(f"❌ API returned empty or invalid response: {response}")
            return False
            
    except Exception as e:
        print(f"❌ API connection test failed: {str(e)}")
        traceback_str = traceback.format_exc()
        print(f"Detailed error information: {traceback_str}")
        return False

# Add summary generation step in the main function
def generate_summaries_for_bookmarks(bookmarks_with_content, model_config=None, force_recompute=False):
    """
    Generates summaries for bookmark content.

    This function iterates through the provided bookmarks with content and generates AI-powered summaries
    using the configured language model. By default, it skips bookmarks that already have a non-empty
    "summary" field to avoid redundant API calls and preserve existing summaries. The force_recompute
    parameter allows overriding this behavior to regenerate all summaries, which is useful for updating
    summaries with improved prompts or models.

    Parameters:
        bookmarks_with_content (list): List of bookmark dictionaries containing content to summarize.
        model_config (ModelConfig, optional): Configuration for the language model. Defaults to environment settings.
        force_recompute (bool): If True, recomputes summaries for all bookmarks regardless of existing summaries.
                                Defaults to False for efficiency.

    Returns:
        list: Updated list of bookmarks with generated summaries.
    """
    if model_config is None:
        model_config = ModelConfig()

    total_count = len(bookmarks_with_content)
    print('Generating summaries for bookmarks...')
    print(f"Using {model_config.model_type} model {model_config.model_name} to generate content summaries for {total_count} items...")
    if force_recompute:
        print("Force recompute mode enabled: regenerating all summaries regardless of existing ones.")

    # First, read the existing file content
    try:
        with open(bookmarks_with_content_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            # Create a map from URL to bookmark for quick lookup of existing summaries
            existing_map = {item.get('url'): item for item in existing_data}
    except (FileNotFoundError, json.JSONDecodeError):
        existing_map = {}
        existing_data = []

    # Use a temporary file to save progress
    temp_file_path = f"{bookmarks_with_content_path}.temp"
    
    # Copy existing data to the temporary file
    try:
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Failed to create temporary file: {str(e)}")
        return existing_data  # Return existing data
    
    success_count = 0
    skipped_count = 0
    for idx, bookmark in enumerate(tqdm(bookmarks_with_content, desc="Summary Generation Progress")):
        url = bookmark["url"]
        title = bookmark["title"]
        print(f"Generating summary [{idx+1}/{total_count}]: {title} - {url}")

        # Check if already processed and has non-empty summary, unless force recompute is enabled
        # This optimization prevents redundant API calls and preserves existing summaries
        existing_summary = existing_map.get(url, {}).get("summary", "").strip()
        if not force_recompute and existing_summary:
            print(f"[{idx+1}/{total_count}] Skipping existing summary: {title} - {url}")
            success_count += 1
            skipped_count += 1
            continue
        
        progress_info = f"[{idx+1}/{total_count}]"
        print(f"{progress_info} Generating summary for the following link: {url}")
        
        # Generate summary
        summary = generate_summary(title, bookmark["content"], url, model_config)
        print(f"{progress_info} title: {title}")
        print(f"{progress_info} summary length: {len(summary)} characters")
        print(f"{progress_info} summary truncated: {summary[:200]}...")
        
        # Add summary to bookmark data
        bookmark["summary"] = summary
        bookmark["summary_model"] = model_config.model_name
        bookmark["summary_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        if "Summary generation failed" not in summary:
            success_count += 1
            print(f"{progress_info} Summary generated successfully")
            
            # Update data structure
            if url in existing_map:
                # Update existing record
                for i, item in enumerate(existing_data):
                    if item.get('url') == url:
                        existing_data[i] = bookmark
                        break
            else:
                # Add new record
                existing_data.append(bookmark)
            
            # Save to temporary file
            try:
                with open(temp_file_path, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, ensure_ascii=False, indent=4)
                # After successfully writing to the temporary file, replace the original file
                os.replace(temp_file_path, bookmarks_with_content_path)
                print(f"{progress_info} Current progress saved")
            except Exception as e:
                print(f"{progress_info} Error saving progress: {str(e)}")
        else:
            print(f"{progress_info} Summary generation failed: {summary}")
        
        # Brief pause after each request to avoid API limits
        time.sleep(0.5)
    
    print(f"Summary generation complete! Success: {success_count}/{total_count}")
    if not force_recompute:
        print(f"Skipped {skipped_count} bookmarks with existing summaries.")
    return existing_data

# Fetch bookmarks using browser_history module
def get_bookmarks(browser=None, profile_path=None):
    """
    Fetches bookmarks from specified browser or all browsers if none specified.

    Parameters:
        browser (str, optional): Browser name (e.g., 'chrome', 'firefox'). If None, fetches from all browsers.
        profile_path (str, optional): Path to browser profile directory.

    Returns:
        list: List of bookmark dictionaries with url, name, date_added, etc.
    """
    urls = []

    try:
        if browser:
            # Map browser name to browser_history class
            browser_map = {
                'chrome': Chrome,
                'firefox': Firefox,
                'edge': Edge,
                'opera': Opera,
                'opera_gx': OperaGX,
                'safari': Safari,
                'vivaldi': Vivaldi,
                'brave': Brave,
            }

            if browser not in browser_map:
                raise ValueError(f"Unsupported browser: {browser}")

            browser_class = browser_map[browser]

            # Initialize browser instance
            if profile_path:
                browser_instance = browser_class(profile_path)
            else:
                browser_instance = browser_class()

            # Fetch bookmarks
            bookmarks_output = browser_instance.fetch_bookmarks()
            bookmarks = bookmarks_output.bookmarks
        else:
            # Fetch from all browsers
            from browser_history import get_bookmarks
            bookmarks_output = get_bookmarks()
            bookmarks = bookmarks_output.bookmarks

        # Convert to the expected format
        for bookmark in bookmarks:
            # browser_history returns tuples of (datetime, url, title, folder)
            timestamp, url, title, folder = bookmark
            bookmark_info = {
                "date_added": timestamp.isoformat() if timestamp else "N/A",
                "date_last_used": "N/A",  # browser_history doesn't provide this
                "guid": "N/A",  # browser_history doesn't provide this
                "id": "N/A",  # browser_history doesn't provide this
                "name": title or "N/A",
                "type": "url",
                "url": url,
            }
            urls.append(bookmark_info)

    except Exception as e:
        print(f"Error fetching bookmarks: {e}")
        # Fallback to empty list or raise error
        raise

    return urls

# Create a session with a retry mechanism
def create_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=3,  # Maximum 3 retries
        backoff_factor=0.5,  # Retry interval backoff factor
        status_forcelist=[429, 500, 502, 503, 504],  # Status codes that trigger a retry
        allowed_methods=["GET"]  # Only retry for GET requests
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

# Clean up text content
def clean_text(text):
    # Remove excessive blank lines and spaces
    lines = [line.strip() for line in text.split('\n')]
    # Filter out empty lines
    lines = [line for line in lines if line]
    # Join lines
    return '\n'.join(lines)

# Initialize Selenium WebDriver
def init_webdriver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Headless mode
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    # Add more user agent information
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36")
    
    # Disable image loading to improve speed
    prefs = {"profile.managed_default_content_settings.images": 2}
    chrome_options.add_experimental_option("prefs", prefs)
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    return driver

# Fetch dynamic content using Selenium
def fetch_with_selenium(url, current_idx=None, total_count=None, title="No Title"):
    """Fetches webpage content using Selenium"""
    progress_info = f"[{current_idx}/{total_count}]" if current_idx and total_count else ""
    
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    # Add a more realistic user agent
    options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36')
    
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        
        print(f"{progress_info} Starting Selenium crawl for: {title} - {url}")
        driver.get(url)
        
        # Wait for page to load
        time.sleep(5)
        
        # Special handling for Zhihu: attempt to close login pop-up if present
        if "zhihu.com" in url:
            try:
                # Attempt to click the close button (multiple possible selectors)
                selectors = ['.Modal-closeButton', '.Button.Modal-closeButton', 
                            'button.Button.Modal-closeButton', '.close']
                for selector in selectors:
                    try:
                        # Use a more robust locator strategy if possible, but stick to the original logic for now
                        close_button = driver.find_element("css selector", selector)
                        close_button.click()
                        print(f"{progress_info} Successfully closed Zhihu login pop-up - using selector: {selector}")
                        time.sleep(1)
                        break
                    except:
                        continue
            except Exception as e:
                print(f"{progress_info} Failed to handle Zhihu login pop-up: {title} - {str(e)}")
        
        # Get page content
        content = driver.page_source
        soup = BeautifulSoup(content, 'html.parser')
        
        # Special handling for Zhihu: extract article content
        if "zhihu.com" in url:
            article = soup.select_one('.Post-RichText') or soup.select_one('.RichText') or soup.select_one('.AuthorInfo') or soup.select_one('article')
            if article:
                text_content = article.get_text(strip=True)
            else:
                text_content = soup.get_text(strip=True)
        else:
            # General webpage handling
            text_content = soup.get_text(strip=True)
        
        # Fix encoding issues
        text_content = fix_encoding(text_content)
        
        # Ensure text is not empty
        if not text_content or len(text_content.strip()) < 5:  # At least 5 characters for valid content
            print(f"{progress_info} Selenium crawl content is empty or too short: {title} - {url}")
            return None
            
        print(f"{progress_info} Selenium successfully crawled: {title} - {url}, content length: {len(text_content)} characters")
        return text_content
        
    except Exception as e:
        print(f"{progress_info} Selenium crawl failed: {title} - {url} - {str(e)}")
        return None
    finally:
        if 'driver' in locals():
            driver.quit()

# Detect and fix encoding issues - Optimized encoding fix function
def fix_encoding(text):
    """
    Detects and fixes text encoding issues, optimized performance version.
    """
    if not text or len(text) < 20:  # Return directly for short text
        return text
    
    # Quick check if fixing is needed - only check a small sample of the text
    sample_size = min(1000, len(text))
    sample_text = text[:sample_size]
    
    # If the proportion of non-ASCII characters in the sample is low, return the original text directly
    non_ascii_count = sum(1 for c in sample_text if ord(c) > 127)
    if non_ascii_count < sample_size * 0.1:  # If non-ASCII characters are less than 10%
        return text
    
    # Check for obvious encoding issue characteristics (consecutive special characters)
    # Use a more efficient method instead of regex
    special_char_sequence = 0
    for c in sample_text:
        if ord(c) > 127:
            special_char_sequence += 1
            if special_char_sequence >= 10:  # Found 10 consecutive non-ASCII characters
                break
        else:
            special_char_sequence = 0
    
    # If there are no obvious encoding issue characteristics, return directly
    if special_char_sequence < 10:
        return text
    
    # Only perform encoding detection on potentially problematic text
    try:
        # Only detect encoding on the sample, not the entire text
        sample_bytes = sample_text.encode('latin-1', errors='ignore')
        detected = chardet.detect(sample_bytes)
        
        # If the detected encoding is different from the current one and confidence is high
        if detected['confidence'] > 0.8 and detected['encoding'] not in ('ascii', 'utf-8'):
            # Re-encode the entire text
            text_bytes = text.encode('latin-1', errors='ignore')
            return text_bytes.decode(detected['encoding'], errors='replace')
    except Exception as e:
        print(f"Encoding fix failed: {e}")
    
    return text

# Crawl webpage content
def fetch_webpage_content(bookmark, current_idx=None, total_count=None):
    """Crawls webpage content"""
    url = bookmark["url"]
    title = bookmark.get("name", "No Title")  # Get title from bookmark
    progress_info = f"[{current_idx}/{total_count}]" if current_idx and total_count else ""
    
    # Initialize variables to prevent unassigned error
    content = None
    crawl_method = None
    
    # Use Selenium directly for Zhihu links
    if "zhihu.com" in url:
        print(f"{progress_info} Detected Zhihu link, using Selenium directly for crawl: {title} - {url}")
        content = fetch_with_selenium(url, current_idx, total_count, title)
        crawl_method = "selenium"
        
        # Record crawl result
        if content:
            print(f"{progress_info} Successfully crawled Zhihu content: {title} - {url}, content length: {len(content)} characters")
        else:
            print(f"{progress_info} Failed to crawl Zhihu content: {title} - {url}")
            return None, {"url": url, "title": title, "reason": "Zhihu content crawl failed", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    else:
        try:
            print(f"{progress_info} Starting crawl: {title} - {url}")
            session = create_session()
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7" # Changed to prioritize English
            }
            response = session.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # Detect response content encoding
            detected_encoding = chardet.detect(response.content)
            if detected_encoding['confidence'] > 0.7:
                response.encoding = detected_encoding['encoding']
            
            # Check content type to ensure it is HTML or text
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' not in content_type.lower() and 'text/plain' not in content_type.lower():
                error_msg = f"Non-text content (Content-Type: {content_type})"
                print(f"{progress_info} Skipping {error_msg}: {title} - {url}")
                failed_info = {"url": url, "title": title, "reason": error_msg, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
                return None, failed_info
                
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract title
            if soup.title:
                title = soup.title.string if soup.title.string else "No Title"
            else:
                title = "No Title"
            
            # Remove unnecessary elements like scripts, styles, navigation, etc.
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'meta', 'link']):
                element.decompose()
            
            # Get the full text content of the page directly
            full_text = soup.get_text(separator='\n')
            
            # Clean up text
            content = clean_text(full_text)
            crawl_method = "requests"
        except Exception as e:
            error_msg = f"Request failed: {str(e)}"
            print(f"{progress_info} {error_msg}: {title} - {url}")
            failed_info = {"url": url, "title": title, "reason": error_msg, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
            return None, failed_info
    
    # If content is empty after regular crawl or for special sites, try Selenium
    if content is None or (isinstance(content, str) and not content.strip()):
        print(f"{progress_info} Regular crawl content is empty, attempting Selenium: {title} - {url}")
        content = fetch_with_selenium(url, current_idx, total_count, title)
        crawl_method = "selenium"
        
        # Record Selenium crawl result
        if content:
            print(f"{progress_info} Selenium successfully crawled {url}, content length: {len(content)} characters")
        else:
            print(f"{progress_info} Selenium crawl failed or content is empty: {url}")
    
    # Fix possible encoding issues
    if title:
        title = fix_encoding(title)
    else:
        title = "No Title"
        
    if content and isinstance(content, str):
        content = fix_encoding(content)
    else:
        content = ""
    
    # Check if content is empty
    if not content or not content.strip():
        error_msg = "Extracted content is empty"
        print(f"{progress_info} {error_msg}: {title} - {url}")
        failed_info = {"url": url, "title": title, "reason": error_msg, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
        return None, failed_info
            
    # Check for content deduplication
    content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
    with content_lock:
        if content_hash in content_hashes:
            print(f"{progress_info} Skipping duplicate content: {title} - {url}")
            return None, None  # Skip saving, but not a failure
        content_hashes.add(content_hash)

    # Create a copy of the bookmark including the content
    bookmark_with_content = bookmark.copy()
    bookmark_with_content["title"] = title
    bookmark_with_content["content"] = content
    bookmark_with_content["content_length"] = len(content)
    bookmark_with_content["crawl_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    bookmark_with_content["crawl_method"] = crawl_method

    print(f"{progress_info} Successfully crawled: {title} - {url}, content length: {len(content)} characters")
    return bookmark_with_content, None

# Parallel crawl bookmark content
def parallel_fetch_bookmarks(bookmarks, max_workers=20, limit=None, flush_interval=60):
    # Determine processing mode based on limit
    if limit:
        print(f"Processing up to {limit} new bookmarks sequentially to accurately enforce limit")
        # Sequential processing for limited crawls
        bookmarks_with_content = []
        failed_records = []
        new_bookmarks_added = 0

        # Initialize flush tracking
        last_flush_time = time.time()

        def flush_to_disk_sequential(current_bookmarks, current_failed):
            """Flush current bookmarks and failed records to disk for sequential processing"""
            try:
                # Read existing bookmarks_with_content.json
                try:
                    with open(bookmarks_with_content_path, 'r', encoding='utf-8') as f:
                        existing_bookmarks = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    existing_bookmarks = []

                # Merge current bookmarks with existing
                merged_bookmarks = existing_bookmarks + current_bookmarks

                # Save atomically using temp file
                temp_file_path = bookmarks_with_content_path + '.temp'
                with open(temp_file_path, 'w', encoding='utf-8') as f:
                    json.dump(merged_bookmarks, f, ensure_ascii=False, indent=4)
                os.replace(temp_file_path, bookmarks_with_content_path)

                # Note: Do not clear lists in sequential mode to maintain return values
            except Exception as e:
                print(f"Error during sequential periodic flush: {e}")

        start_time = time.time()
        print(f"Starting sequential crawl of bookmark content")
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        for idx, bookmark in enumerate(bookmarks):
            if new_bookmarks_added >= limit:
                print(f"Reached limit of {limit} new bookmarks added")
                break

            url = bookmark['url']
            url_hash = hashlib.sha256(url.encode('utf-8')).hexdigest()
            if url_hash in url_hashes:
                title = bookmark.get("name", "No Title")
                print(f"Skipping duplicate URL [{idx+1}]: {title} - {url}")
                continue  # Skip duplicates without counting towards limit
            url_hashes.add(url_hash)

            title = bookmark.get("name", "No Title")
            print(f"Processing bookmark [{idx+1}]: {title} - {url}")

            result, failed_info = fetch_webpage_content(bookmark, idx+1, None)  # No total_count for sequential
            if result:
                bookmarks_with_content.append(result)
                new_bookmarks_added += 1
                print(f"Successfully added bookmark {new_bookmarks_added}/{limit}")

                # Check for periodic flush
                current_time = time.time()
                if current_time - last_flush_time >= flush_interval:
                    print(f"Flush interval ({flush_interval}s) reached, flushing to disk...")
                    flush_to_disk_sequential(bookmarks_with_content, failed_records)
                    print("Intermediate flush complete.")
                    last_flush_time = current_time

            if failed_info:
                failed_records.append(failed_info)

        end_time = time.time()
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Print elapsed time information
        elapsed_time = end_time - start_time
        elapsed_minutes = elapsed_time / 60
        if elapsed_time > 60:
            print(f"Total time for sequential bookmark crawl: {elapsed_minutes:.2f} minutes ({elapsed_time:.2f} seconds)")
        else:
            print(f"Total time for sequential bookmark crawl: {elapsed_time:.2f} seconds")

        # Calculate average processing time per bookmark
        processed_count = idx + 1
        if processed_count > 0:
            avg_time_per_bookmark = elapsed_time / processed_count
            print(f"Average processing time per bookmark: {avg_time_per_bookmark:.2f} seconds")

        # Force final flush after all processing is complete
        if bookmarks_with_content or failed_records:
            print("Performing final flush to disk...")
            flush_to_disk_sequential(bookmarks_with_content, failed_records)
            print("Final flush complete.")

        return bookmarks_with_content, failed_records, new_bookmarks_added
    else:
        # Original parallel processing for unlimited crawls
        print(f"Processing all {len(bookmarks)} bookmarks in parallel")
        bookmarks_to_process = bookmarks

        bookmarks_with_content = []
        failed_records = []
        skipped_url_count = 0

        # Batch flushing variables for thread-safety
        flush_lock = threading.Lock()
        bookmarks_lock = threading.Lock()
        last_flush_time = time.time()
        flush_in_progress = False
        flush_flag_lock = threading.Lock()

        def flush_to_disk(current_bookmarks, current_failed):
            nonlocal flush_in_progress
            with flush_flag_lock:
                if flush_in_progress:
                    return  # Prevent overlapping flushes
                flush_in_progress = True
            try:
                # Read existing bookmarks_with_content.json
                try:
                    with open(bookmarks_with_content_path, 'r', encoding='utf-8') as f:
                        existing_bookmarks = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    existing_bookmarks = []

                # Merge current bookmarks with existing
                merged_bookmarks = existing_bookmarks + current_bookmarks

                # Save atomically using temp file
                temp_file_path = bookmarks_with_content_path + '.temp'
                with open(temp_file_path, 'w', encoding='utf-8') as f:
                    json.dump(merged_bookmarks, f, ensure_ascii=False, indent=4)
                os.replace(temp_file_path, bookmarks_with_content_path)

                # Clear the current lists after successful flush
                current_bookmarks.clear()
                current_failed.clear()
            except Exception as e:
                print(f"Error during periodic flush: {e}")
            finally:
                with flush_flag_lock:
                    flush_in_progress = False

        def monitor_thread():
            nonlocal last_flush_time, bookmarks_with_content, failed_records
            while True:
                time.sleep(1)  # Check every second
                current_time = time.time()
                with bookmarks_lock:
                    if current_time - last_flush_time >= flush_interval:
                        # Trigger flush when interval has passed
                        print(f"Flush interval ({flush_interval}s) reached, flushing to disk...")
                        flush_to_disk(bookmarks_with_content, failed_records)
                        print("Intermediate flush complete.")
                        last_flush_time = current_time

        # Start daemon thread to monitor counter and trigger flushes
        # Treats persistence as a "sidecar" process, similar to event-sourcing in databases.
        monitor = threading.Thread(target=monitor_thread, daemon=True)
        monitor.start()

        # Use ThreadPoolExecutor for parallel crawling of bookmark content
        start_time = time.time()
        total_count = len(bookmarks_to_process)
        print(f"Starting parallel crawl of bookmark content, max workers: {max_workers}, total: {total_count}")
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Create a list to store all tasks
        futures = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            for idx, bookmark in enumerate(bookmarks_to_process):
                url = bookmark['url']
                url_hash = hashlib.sha256(url.encode('utf-8')).hexdigest()
                if url_hash in url_hashes:
                    title = bookmark.get("name", "No Title")
                    print(f"Skipping duplicate URL [{idx+1}/{total_count}]: {title} - {url}")
                    skipped_url_count += 1
                    continue
                url_hashes.add(url_hash)

                # Print progress before submitting the task
                title = bookmark.get("name", "No Title")
                print(f"Submitting task [{idx+1}/{total_count}]: {title} - {bookmark['url']}")
                future = executor.submit(fetch_webpage_content, bookmark, idx+1, total_count)
                futures.append(future)

            # Use tqdm to create a progress bar
            for future in tqdm(futures, total=len(futures), desc="Crawl Progress"):
                result, failed_info = future.result()
                with bookmarks_lock:
                    if result:
                        bookmarks_with_content.append(result)
                    if failed_info:
                        failed_records.append(failed_info)

        end_time = time.time()
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Print elapsed time information
        elapsed_time = end_time - start_time
        elapsed_minutes = elapsed_time / 60
        if elapsed_time > 60:
            print(f"Total time for parallel bookmark crawl: {elapsed_minutes:.2f} minutes ({elapsed_time:.2f} seconds)")
        else:
            print(f"Total time for parallel bookmark crawl: {elapsed_time:.2f} seconds")

        # Calculate average processing time per bookmark
        if total_count > 0:
            avg_time_per_bookmark = elapsed_time / total_count
            print(f"Average processing time per bookmark: {avg_time_per_bookmark:.2f} seconds")

        # Force final flush after all processing is complete
        with bookmarks_lock:
            if bookmarks_with_content or failed_records:
                print("Performing final flush to disk...")
                flush_to_disk(bookmarks_with_content, failed_records)
                print("Final flush complete.")

        return bookmarks_with_content, failed_records, len(bookmarks_with_content)

# Parse command-line arguments
def parse_args():
    # Create the argument parser with a clear description of the application's purpose.
    parser = argparse.ArgumentParser(description='Crawl browser bookmarks and build a knowledge base')

    # Argument for limiting the number of bookmarks to process.
    parser.add_argument('--limit', type=int, help='Limit the number of bookmarks to process (0 for no limit)')

    # Argument for setting the number of concurrent workers for parallel fetching.
    parser.add_argument('--workers', type=int, help='Number of worker threads for parallel fetching')

    # Flag to skip the summary generation step, useful for content fetching only.
    parser.add_argument('--no-summary', action='store_true', help='Skip the summary generation step')

    # Flag to generate summaries from an existing content file, skipping the crawl.
    parser.add_argument('--from-json', action='store_true', help='Generate summaries from existing bookmarks_with_content.json')

    # Add optional command-line argument to specify a custom browser.
    # This allows the application to read bookmarks from a specific browser.
    parser.add_argument(
        '--browser',
        '-b',
        type=str,
        choices=['chrome', 'firefox', 'edge', 'opera', 'opera_gx', 'safari', 'vivaldi', 'brave'],
        help='Specify the browser to fetch bookmarks from. If not specified, fetches from all browsers.'
    )

    # Add optional command-line argument to specify a custom profile path.
    # This allows the application to read bookmarks from a specific profile directory.
    parser.add_argument(
        '--profile-path',
        type=str,
        help='Specify a custom path to the browser profile directory. Used in conjunction with --browser.'
    )

    # Add optional command-line argument to specify a custom config file path.
    parser.add_argument(
        '--config',
        type=str,
        default='default_config.toml',
        help='Path to the TOML configuration file (default: default_config.toml)'
    )

    # Add --rebuild argument to rebuild the entire index from scratch
    parser.add_argument(
        '--rebuild',
        action='store_true',
        help='Rebuild the entire index from scratch instead of resuming from existing bookmarks_with_content.json'
    )

    # Add --flush-interval argument to control the interval for flushing to disk
    parser.add_argument(
        '--flush-interval',
        type=int,
        default=60,
        help='Interval in seconds for flushing to disk to save intermediate results (default: 60)'
    )

    # Add --force-recompute-summaries argument to force regeneration of all summaries
    parser.add_argument(
        '--force-recompute-summaries',
        action='store_true',
        help='Force recomputation of summaries for all bookmarks, overriding the default skip behavior for existing summaries'
    )
    return parser.parse_args()

# Main function to orchestrate the bookmark crawling and summarization process.
def main():
    # Parse command-line arguments
    args = parse_args()

    # Load TOML configuration
    config_data = load_config(args.config)

    # Read configuration from TOML file, command-line arguments take precedence
    bookmark_limit = args.limit if args.limit is not None else 0  # Default: no limit
    max_workers = args.workers if args.workers is not None else 20  # Default: 20 worker threads
    generate_summary_flag = not args.no_summary  # Command-line flag overrides config
    flush_interval = args.flush_interval  # Interval for flushing to disk

    # Load existing bookmarks_with_content.json if not rebuilding from scratch
    existing_bookmarks = []
    if not args.rebuild:
        try:
            with open(bookmarks_with_content_path, 'r', encoding='utf-8') as f:
                existing_bookmarks = json.load(f)
            print(f"Loaded {len(existing_bookmarks)} existing bookmarks from {bookmarks_with_content_path}")

            # Populate global deduplication sets with existing data
            global url_hashes, content_hashes
            for bookmark in existing_bookmarks:
                url = bookmark.get('url')
                if url:
                    url_hash = hashlib.sha256(url.encode('utf-8')).hexdigest()
                    url_hashes.add(url_hash)
                content = bookmark.get('content')
                if content:
                    content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
                    content_hashes.add(content_hash)
            print(f"Populated deduplication sets: {len(url_hashes)} URLs, {len(content_hashes)} content hashes")
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"No existing {bookmarks_with_content_path} found or invalid JSON, starting fresh")
            existing_bookmarks = []
    else:
        print("Rebuilding from scratch (--rebuild flag used)")

    # If the --from-json argument is used, read directly from the JSON file and generate summaries
    if args.from_json:
        print("Generating summaries from existing bookmarks_with_content.json...")
        try:
            with open(bookmarks_with_content_path, 'r', encoding='utf-8') as f:
                bookmarks_with_content = json.load(f)

            if not bookmarks_with_content:
                print("Error: bookmarks_with_content.json is empty or incorrectly formatted")
                return

            if bookmark_limit > 0:
                print(f"Processing only the first {bookmark_limit} bookmarks based on limit")
                bookmarks_with_content = bookmarks_with_content[:bookmark_limit]

            # Configure model and generate summaries
            model_config = ModelConfig(config_data)

            # Test API connection
            if not test_api_connection(model_config):
                print("LLM API connection failed, please check configuration and try again.", model_config.api_base, model_config.model_name, model_config.api_key, model_config.model_type)
                return

            # Generate summaries for content, respecting the force recompute flag
            bookmarks_with_content = generate_summaries_for_bookmarks(bookmarks_with_content, model_config, args.force_recompute_summaries)

            # Save the updated content using atomic temp file
            temp_file_path = f"{bookmarks_with_content_path}.temp"
            try:
                with open(temp_file_path, "w", encoding="utf-8") as output_file:
                    json.dump(bookmarks_with_content, output_file, ensure_ascii=False, indent=4)
                # Atomic replace
                os.replace(temp_file_path, bookmarks_with_content_path)
                print(f"Successfully saved {len(bookmarks_with_content)} bookmarks to {bookmarks_with_content_path}")
            except Exception as e:
                print(f"Error saving bookmarks: {str(e)}")
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                raise

            print(f"Summary generation complete, {bookmarks_with_content_path} updated")
            return

        except FileNotFoundError:
            print(f"Error: File not found {bookmarks_with_content_path}")
            return
        except json.JSONDecodeError:
            print(f"Error: {bookmarks_with_content_path} is not a valid JSON file")
            return
        except Exception as e:
            print(f"Error processing JSON file: {str(e)}")
            return

    # Original crawling logic
    print(f"Configuration:")
    print(f"  - Browser: {args.browser if args.browser else 'All browsers'}")
    print(f"  - Profile Path: {args.profile_path if args.profile_path else 'Default'}")
    print(f"  - Bookmark Limit: {bookmark_limit if bookmark_limit > 0 else 'No Limit'}")
    print(f"  - Parallel Workers: {max_workers}")
    print(f"  - Generate Summary: {'Yes' if generate_summary_flag else 'No'}")

    # Get bookmark data
    bookmarks = get_bookmarks(browser=args.browser, profile_path=args.profile_path)

    # Filter bookmarks: remove empty URLs, 10.0. network URLs, and non-qualifying types
    filtered_bookmarks = []
    for bookmark in bookmarks:
        url = bookmark["url"]
        # Check for empty URL, URL type, not "Extension" name, and not 10.0. network URL
        if (url and
            bookmark["type"] == "url" and
            bookmark["name"] != "扩展程序" and # "扩展程序" is a folder name for extensions in Chinese Chrome
            not re.match(r"https?://10\.0\.", url)):
            # If not rebuilding, skip URLs already in existing bookmarks
            if not args.rebuild:
                url_hash = hashlib.sha256(url.encode('utf-8')).hexdigest()
                if url_hash in url_hashes:
                    print(f"Skipping already indexed URL: {bookmark.get('name', 'No Title')} - {url}")
                    continue
            filtered_bookmarks.append(bookmark)
    
    # Save filtered bookmark data
    with open(bookmarks_path, "w", encoding="utf-8") as output_file:
        json.dump(filtered_bookmarks, output_file, ensure_ascii=False, indent=4)
    
    # Parallel crawl bookmark content
    bookmarks_with_content, failed_records, skipped_url_count = parallel_fetch_bookmarks(
        filtered_bookmarks,
        max_workers=max_workers,
        limit=bookmark_limit if bookmark_limit > 0 else None,
        flush_interval=flush_interval
    )
    
    # Only execute the following code if summary generation is enabled
    if generate_summary_flag and bookmarks_with_content:
        # Configure model
        model_config = ModelConfig(config_data)

        # Test API connection
        if not test_api_connection(model_config):
            print("LLM API connection failed, please check configuration and try again.", model_config.api_base, model_config.model_name, model_config.api_key, model_config.model_type)
            print("Skipping summary generation step...")
        else:
            # Generate summaries for the crawled content, respecting the force recompute flag
            bookmarks_with_content = generate_summaries_for_bookmarks(bookmarks_with_content, model_config, args.force_recompute_summaries)
    elif not generate_summary_flag:
        print("Skipping summary generation step based on configuration...")

    # Merge new crawled results with existing data if not rebuilding
    if not args.rebuild:
        bookmarks_with_content = existing_bookmarks + bookmarks_with_content
        print(f"Merged {len(existing_bookmarks)} existing bookmarks with {len(bookmarks_with_content) - len(existing_bookmarks)} new bookmarks")

    # Save bookmark data with content using atomic temp file
    temp_file_path = f"{bookmarks_with_content_path}.temp"
    try:
        with open(temp_file_path, "w", encoding="utf-8") as output_file:
            json.dump(bookmarks_with_content, output_file, ensure_ascii=False, indent=4)
        # Atomic replace
        os.replace(temp_file_path, bookmarks_with_content_path)
        print(f"Successfully saved {len(bookmarks_with_content)} bookmarks to {bookmarks_with_content_path}")
    except Exception as e:
        print(f"Error saving bookmarks: {str(e)}")
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise
    
    # Save failed URLs and reasons
    with open(failed_urls_path, "w", encoding="utf-8") as f:
        json.dump(failed_records, f, ensure_ascii=False, indent=4)
    
    print(f"Extracted {len(filtered_bookmarks)} valid bookmarks, saved to {bookmarks_path}")
    print(f"Successfully crawled content for {len(bookmarks_with_content)} bookmarks, saved to {bookmarks_with_content_path}")
    print(f"Skipped {skipped_url_count} duplicate URLs during crawling")
    print(f"Failed to crawl {len(failed_records)} URLs, details saved to {failed_urls_path}")
    
    # Print list of failed URLs and titles for easy viewing
    if failed_records:
        print("\nFailed URLs and Titles:")
        for idx, record in enumerate(failed_records):
            print(f"{idx+1}. {record.get('title', 'No Title')} - {record['url']} - Reason: {record['reason']}")
    
    # Display content length statistics
    if bookmarks_with_content:
        total_length = sum(b.get("content_length", 0) for b in bookmarks_with_content)
        avg_length = total_length / len(bookmarks_with_content)
        print(f"Average crawled content length: {avg_length:.2f} characters")
        print(f"Longest content: {max(b.get('content_length', 0) for b in bookmarks_with_content)} characters")
        print(f"Shortest content: {min(b.get('content_length', 0) for b in bookmarks_with_content)} characters")
        
        # Statistics on crawl methods used
        selenium_count = sum(1 for b in bookmarks_with_content if b.get("crawl_method") == "selenium")
        requests_count = sum(1 for b in bookmarks_with_content if b.get("crawl_method") == "requests")
        print(f"Crawled using Selenium: {selenium_count} items")
        print(f"Crawled using Requests: {requests_count} items")

# This function is redundant as its logic is mostly covered by fetch_with_selenium, 
# but it was present in the original file. I will translate it and keep it for completeness, 
# but it is not called in main().
def fetch_zhihu_content(url, current_idx=None, total_count=None, title="No Title"):
    """Specifically handles Zhihu links"""
    progress_info = f"[{current_idx}/{total_count}]" if current_idx and total_count else ""
    
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    # Add a more realistic user agent
    options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36')
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    
    try:
        print(f"{progress_info} Using dedicated method to crawl Zhihu content: {title} - {url}")
        driver.get(url)
        # Wait for page to load
        time.sleep(3)
        
        # Detect and close login pop-up
        try:
            # Note: find_element_by_css_selector is deprecated, but keeping the original logic structure
            login_close = driver.find_element("css selector", '.Modal-closeButton')
            login_close.click()
            print(f"{progress_info} Successfully closed Zhihu login pop-up")
            time.sleep(1)
        except Exception as e:
            print(f"{progress_info} Failed to close Zhihu login pop-up or no need to close: {title} - {str(e)}")
        
        # Get page content
        content = driver.page_source
        soup = BeautifulSoup(content, 'html.parser')
        
        # Extract main content
        article = soup.select_one('.Post-RichText') or soup.select_one('.RichText')
        if article:
            result = article.get_text()
            print(f"{progress_info} Successfully extracted Zhihu article content: {title}, length: {len(result)} characters")
            return result
        else:
            result = soup.get_text()
            print(f"{progress_info} Zhihu article body not found, using full text: {title}, length: {len(result)} characters")
            return result
    
    except Exception as e:
        print(f"{progress_info} Zhihu crawl exception: {title} - {url} - {str(e)}")
        return None
    finally:
        driver.quit()

if __name__ == "__main__":
    main()