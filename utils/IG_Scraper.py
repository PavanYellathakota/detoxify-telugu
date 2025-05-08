# ============================================================================ #
#                          📸  INSTAGRAM SCRAPER                               #
# ============================================================================ #
# Filename     : IG_Scrapper.py
# Description  : Scrapes public Instagram data (profiles, posts, hashtags) using Selenium
#                with a Streamlit UI for user interaction.
# Author       : Custom Development (xAI)
# Created Date : May 06, 2025
# Project      : Social Media Data Collection
# ============================================================================ #

# Importing necessary libraries
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import pandas as pd
import streamlit as st
import time
import random
import urllib.parse
import os

def resolve_instagram_url(url):
    """
    Resolves shortened or malformed Instagram URLs to canonical format.
    Args:
        url (str): Input Instagram URL.
    Returns:
        str: Canonical Instagram URL.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc == "www.instagram.com" or parsed.netloc == "instagram.com":
        path = parsed.path.strip("/")
        return f"https://www.instagram.com/{path.split('/')[0]}/"
    return url

def scrape_instagram_data(urls, output_file, max_posts=50):
    """
    Scrapes public Instagram data from provided URLs using Selenium.
    Args:
        urls (list): List of Instagram profile or hashtag URLs.
        output_file (str): Path to save the scraped data CSV.
        max_posts (int): Maximum number of posts to scrape per profile/hashtag.
    Returns:
        pd.DataFrame: DataFrame of scraped data, or None if no data collected.
    """
    # Validating ChromeDriver path
    chrome_driver_path = r"C:\Users\prudh\Downloads\chromedriver-win64\chromedriver-win64\chromedriver.exe"
    if not os.path.isfile(chrome_driver_path):
        raise FileNotFoundError(f"ChromeDriver not found at {chrome_driver_path}. Please verify the path.")

    st.info(f"Using ChromeDriver at: {chrome_driver_path}")

    # Configuring Selenium options for headless browsing
    chrome_options = Options()
    # chrome_options.add_argument("--headless")  # Uncomment for production
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36")

    try:
        driver = webdriver.Chrome(
            service=Service(chrome_driver_path),
            options=chrome_options
        )
    except Exception as e:
        raise Exception(f"Failed to start ChromeDriver: {e}")

    all_data = []

    for url in urls:
        # Resolving URL
        canonical_url = resolve_instagram_url(url)
        st.write(f"Scraping data for: {canonical_url}")
        driver.get(canonical_url)
        time.sleep(5)  # Initial wait for page load

        try:
            # Scroll to load content
            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(2)

            # Waiting for profile or hashtag page to load
            st.write("Waiting for Instagram page...")
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.CLASS_NAME, "_aagw"))
            )
            st.success("Page content found.")

            # Extract profile metadata (if profile page)
            try:
                username_elem = driver.find_element(By.CLASS_NAME, "_aacl")
                username = username_elem.text.strip()
                follower_elem = driver.find_element(By.XPATH, "//span[text()='followers']//following::span[1]")
                followers = follower_elem.text.strip().replace(",", "")
                post_count_elem = driver.find_element(By.CLASS_NAME, "_ac2a")
                post_count = post_count_elem.text.strip().replace(",", "")
                st.write(f"Profile: {username}, Followers: {followers}, Posts: {post_count}")
            except NoSuchElementException:
                st.warning("Not a profile page or metadata not found. Assuming hashtag or post page.")
                username, followers, post_count = "unknown", "unknown", "unknown"

            # Scrolling to load posts
            scroll_attempts = 0
            max_scroll_attempts = 20
            last_post_count = 0
            while scroll_attempts < max_scroll_attempts:
                driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
                time.sleep(random.uniform(2, 4))
                scroll_attempts += 1
                post_elements = driver.find_elements(By.CLASS_NAME, "_aabd")
                current_count = len(post_elements)
                st.write(f"Scroll attempt {scroll_attempts}, found {current_count} post elements")
                if current_count >= max_posts or current_count == last_post_count:
                    break
                last_post_count = current_count

            # Extracting post details
            st.write("Extracting posts...")
            post_elements = driver.find_elements(By.CLASS_NAME, "_aabd")
            st.write(f"Found {len(post_elements)} post elements.")

            if len(post_elements) == 0:
                st.error("No post elements detected. Likely blocked or page issue.")
                continue

            for i, post_elem in enumerate(post_elements[:max_posts]):
                try:
                    # Open post in modal
                    driver.execute_script("arguments[0].click();", post_elem)
                    time.sleep(2)  # Wait for modal to load

                    # Extract post data
                    caption_elem = driver.find_elements(By.CLASS_NAME, "_aacl")
                    caption = caption_elem[1].text.strip() if len(caption_elem) > 1 else "No caption"
                    likes_elem = driver.find_element(By.CLASS_NAME, "_ae2s")
                    likes = likes_elem.text.strip().replace(",", "")
                    comments_elem = driver.find_element(By.CLASS_NAME, "_ae3z")
                    comments = comments_elem.text.strip().replace(",", "")

                    post_data = {
                        "url": canonical_url,
                        "username": username,
                        "followers": followers,
                        "post_count": post_count,
                        "post_index": i + 1,
                        "caption": caption,
                        "likes": likes,
                        "comments_count": comments
                    }
                    all_data.append(post_data)
                    st.write(f"Collected post {i+1}: {caption[:50]}... (Likes: {likes}, Comments: {comments})")

                    # Close modal
                    driver.find_element(By.CLASS_NAME, "_a9-z").click()
                    time.sleep(1)

                except NoSuchElementException as e:
                    st.error(f"Error processing post {i+1}: Missing element - {e}")
                    driver.find_element(By.CLASS_NAME, "_a9-z").click() if "modal" in driver.current_url else None
                    time.sleep(1)
                    continue
                except Exception as e:
                    st.error(f"Error processing post {i+1}: {e}")
                    driver.find_element(By.CLASS_NAME, "_a9-z").click() if "modal" in driver.current_url else None
                    time.sleep(1)
                    continue

        except TimeoutException:
            st.error(f"Timeout loading {canonical_url}. Possible rate limit or block.")
        except Exception as e:
            st.error(f"Error scraping {canonical_url}: {e}")

    # Saving data to CSV
    driver.quit()
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(output_file, index=False, encoding="utf-8")
        st.success(f"Saved {len(all_data)} posts to {output_file}")
        return df
    else:
        st.warning("No data collected for any URLs.")
        return None

def render_data_collection_ui(data_path):
    """
    Renders the Streamlit UI for Instagram scraping.
    Args:
        data_path (str): Path to the Instagram data CSV for saving and appending.
    """
    st.markdown("<h2>📸 Instagram Scraper</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:16px;'>
    Enter Instagram profile or hashtag URLs to scrape public data for analysis.
    The scraped data will be saved to the Instagram dataset.
    </div>
    """, unsafe_allow_html=True)

    # Input for Instagram URLs
    urls_input = st.text_area(
        "Enter Instagram URLs (one per line)",
        placeholder="https://www.instagram.com/username/\nhttps://www.instagram.com/explore/tags/hashtag/",
        height=150
    )

    # Input for maximum posts
    max_posts = st.number_input(
        "Maximum Posts per Profile/Hashtag",
        min_value=10,
        max_value=500,
        value=50,
        step=10
    )

    # Using the provided data_path as the output file
    output_file = data_path

    # Ensuring output directory exists
    output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else "."
    os.makedirs(output_dir, exist_ok=True)

    # Scrape button
    if st.button("Start Scraping"):
        if not urls_input.strip():
            st.error("Please provide at least one Instagram URL.")
            return

        # Processing URLs
        urls = [url.strip() for url in urls_input.split("\n") if url.strip()]
        st.write(f"Found {len(urls)} URLs to scrape.")

        # Running scraper
        with st.spinner("Scraping Instagram data... This may take a while."):
            scraped_df = scrape_instagram_data(urls, output_file, max_posts)

        # Displaying results
        if scraped_df is not None:
            st.dataframe(scraped_df, use_container_width=True)
            st.markdown(f"**Total Posts Scraped**: {len(scraped_df)}")

            # Option to append to existing dataset
            append_to_existing = st.checkbox("Append scraped data to existing dataset?", value=False)
            if append_to_existing:
                try:
                    existing_df = pd.read_csv(data_path) if os.path.exists(data_path) else pd.DataFrame(columns=["url", "username", "followers", "post_count", "post_index", "caption", "likes", "comments_count"])
                    updated_df = pd.concat([existing_df, scraped_df], ignore_index=True)
                    updated_df.to_csv(data_path, index=False, encoding="utf-8")
                    st.success(f"Appended {len(scraped_df)} posts to {data_path}")
                except Exception as e:
                    st.error(f"Error appending to dataset: {e}")