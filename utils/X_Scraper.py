# ============================================================================ #
#                          🐦  X (TWITTER) POSTS SCRAPER                      #
# ============================================================================ #
# Filename     : X_Scraper.py
# Description  : Scrapes X (Twitter) posts using Selenium with a Streamlit UI for
#                user interaction. Extracts post text, likes, retweets, and comments.
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

def resolve_x_url(url):
    """
    Resolves malformed X URLs to canonical format.
    Args:
        url (str): Input X URL.
    Returns:
        str: Canonical X URL.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc in ["twitter.com", "x.com"]:
        path = parsed.path.strip("/")
        return f"https://x.com/{path.split('/')[0]}/"
    return url

def scrape_x_posts_selenium(urls, output_file, max_posts=50):
    """
    Scrapes X posts from provided profile or hashtag URLs using Selenium.
    Args:
        urls (list): List of X profile or hashtag URLs.
        output_file (str): Path to save the scraped posts CSV.
        max_posts (int): Maximum number of posts to scrape per profile/hashtag.
    Returns:
        pd.DataFrame: DataFrame of scraped posts, or None if no posts collected.
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

    all_posts = []

    for url in urls:
        # Resolving URL
        canonical_url = resolve_x_url(url)
        st.write(f"Scraping posts for: {canonical_url}")
        driver.get(canonical_url)
        time.sleep(5)  # Initial wait for page load

        try:
            # Scroll to load content
            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(2)

            # Waiting for timeline to load
            st.write("Waiting for X timeline...")
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "article"))
            )
            st.success("Timeline content found.")

            # Extract profile metadata (if profile page)
            try:
                username_elem = driver.find_element(By.CSS_SELECTOR, "[data-testid='UserName']")
                username = username_elem.text.strip().split('\n')[0]
                follower_elem = driver.find_element(By.CSS_SELECTOR, "[data-testid='UserProfileHeader_Items'] span")
                followers = follower_elem.text.strip().replace(" Followers", "").replace(",", "")
                st.write(f"Profile: {username}, Followers: {followers}")
            except NoSuchElementException:
                st.warning("Not a profile page or metadata not found. Assuming hashtag or search page.")
                username, followers = "unknown", "unknown"

            # Scrolling to load posts dynamically
            scroll_attempts = 0
            max_scroll_attempts = 20
            last_post_count = 0
            while scroll_attempts < max_scroll_attempts:
                driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
                time.sleep(random.uniform(2, 4))
                scroll_attempts += 1
                post_elements = driver.find_elements(By.CSS_SELECTOR, "article")
                current_count = len(post_elements)
                st.write(f"Scroll attempt {scroll_attempts}, found {current_count} post elements")
                if current_count >= max_posts or current_count == last_post_count:
                    break
                last_post_count = current_count

            # Extracting post details
            st.write("Extracting posts...")
            post_elements = driver.find_elements(By.CSS_SELECTOR, "article")
            st.write(f"Found {len(post_elements)} post elements.")

            if len(post_elements) == 0:
                try:
                    timeline = driver.find_element(By.CSS_SELECTOR, "[data-testid='primaryColumn']")
                    html_snippet = timeline.get_attribute("innerHTML")[:1000]
                    st.write(f"Timeline HTML snippet: {html_snippet}")
                except Exception as e:
                    st.error(f"Couldn’t get HTML: {e}")
                st.error("No post elements detected. Likely blocked or selector issue.")
                continue

            for i, post_elem in enumerate(post_elements[:max_posts]):
                try:
                    # Extract post text
                    text_elem = post_elem.find_element(By.CSS_SELECTOR, "[data-testid='tweetText']")
                    post_text = text_elem.text.strip()

                    # Extract stats (likes, retweets, comments)
                    stats = post_elem.find_elements(By.CSS_SELECTOR, "[data-testid='like'], [data-testid='retweet'], [data-testid='reply']")
                    likes = stats[0].text.strip().replace(",", "") if stats[0].text else "0"
                    retweets = stats[1].text.strip().replace(",", "") if stats[1].text else "0"
                    comments = stats[2].text.strip().replace(",", "") if stats[2].text else "0"

                    post_data = {
                        "url": canonical_url,
                        "username": username,
                        "followers": followers,
                        "post_index": i + 1,
                        "post_text": post_text,
                        "likes": likes,
                        "retweets": retweets,
                        "comments_count": comments
                    }
                    all_posts.append(post_data)
                    st.write(f"Collected post {i+1}: {post_text[:50]}... (Likes: {likes}, Retweets: {retweets}, Comments: {comments})")

                except NoSuchElementException as e:
                    st.error(f"Error processing post {i+1}: Missing element - {e}")
                    continue
                except Exception as e:
                    st.error(f"Error processing post {i+1}: {e}")
                    continue

        except TimeoutException:
            st.error(f"Timeout loading {canonical_url}. Possible rate limit or block.")
        except Exception as e:
            st.error(f"Error scraping {canonical_url}: {e}")

    # Saving posts to CSV
    driver.quit()
    if all_posts:
        df = pd.DataFrame(all_posts)
        df.to_csv(output_file, index=False, encoding="utf-8")
        st.success(f"Saved {len(all_posts)} posts to {output_file}")
        return df
    else:
        st.warning("No posts collected for any URLs.")
        return None

def render_data_collection_ui(data_path):
    """
    Renders the Streamlit UI for X posts scraping.
    Args:
        data_path (str): Path to the X data CSV for saving and appending.
    """
    st.markdown("<h2>🐦 X (Twitter) Posts Scraper</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:16px;'>
    Enter X profile or hashtag URLs to scrape posts for analysis. The scraped posts
    will be saved to the X dataset and can be appended to the existing dataset.
    </div>
    """, unsafe_allow_html=True)

    # Input for X URLs
    urls_input = st.text_area(
        "Enter X Profile or Hashtag URLs (one per line)",
        placeholder="https://x.com/username\nhttps://x.com/hashtag/hashtag",
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
            st.error("Please provide at least one X URL.")
            return

        # Processing URLs
        urls = [url.strip() for url in urls_input.split("\n") if url.strip()]
        st.write(f"Found {len(urls)} URLs to scrape.")

        # Running scraper
        with st.spinner("Scraping X posts... This may take a while."):
            scraped_df = scrape_x_posts_selenium(urls, output_file, max_posts)

        # Displaying results
        if scraped_df is not None:
            st.dataframe(scraped_df, use_container_width=True)
            st.markdown(f"**Total Posts Scraped**: {len(scraped_df)}")

            # Option to append to existing dataset
            append_to_existing = st.checkbox("Append scraped posts to existing dataset?", value=False)
            if append_to_existing:
                try:
                    existing_df = pd.read_csv(data_path) if os.path.exists(data_path) else pd.DataFrame(columns=["url", "username", "followers", "post_index", "post_text", "likes", "retweets", "comments_count"])
                    updated_df = pd.concat([existing_df, scraped_df], ignore_index=True)
                    updated_df.to_csv(data_path, index=False, encoding="utf-8")
                    st.success(f"Appended {len(scraped_df)} posts to {data_path}")
                except Exception as e:
                    st.error(f"Error appending to dataset: {e}")