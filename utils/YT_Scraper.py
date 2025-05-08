# ============================================================================ #
#                          📺  YOUTUBE COMMENTS SCRAPER                       #
# ============================================================================ #
# Filename     : YT_Scraper.py
# Description  : Scrapes YouTube comments using GeckoDriver (Firefox) with a Streamlit UI.
# Author       : PAVAN YELLATHAKOTA (pye.pages.dev)
# Created Date : APR 2025
# Project      : Toxicity Detection / Classification Platform
# ============================================================================ #

# Importing necessary libraries
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException
import pandas as pd
import streamlit as st
import time
import random
import urllib.parse
import os

def resolve_youtube_url(url):
    """
    Resolves shortened YouTube URLs (e.g., youtu.be) to canonical format.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc == "youtu.be":
        video_id = parsed.path.lstrip("/")
        return f"https://www.youtube.com/watch?v={video_id}"
    return url

def scrape_youtube_comments_selenium(urls, output_file, max_comments=100):
    """
    Scrapes YouTube comments using GeckoDriver (Firefox).
    """
    # Specify the path to GeckoDriver
    geckodriver_path = r"C:\Users\prudh\Desktop\Toxicity_Platform\assets\webdriver\geckodriver.exe"  # Updated path with drive letter
    if not os.path.isfile(geckodriver_path):
        raise FileNotFoundError(f"GeckoDriver not found at {geckodriver_path}. Please download a version compatible with Firefox 138.0.1 from https://github.com/mozilla/geckodriver/releases (e.g., v0.37.0 or later) and place it in this directory.")

    st.info(f"Using GeckoDriver at: {geckodriver_path}")

    # Configure Firefox options
    firefox_options = Options()
    # firefox_options.add_argument("--headless")  # Uncomment for production
    firefox_options.add_argument("--disable-gpu")
    firefox_options.add_argument("--no-sandbox")
    firefox_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:138.0) Gecko/20100101 Firefox/138.0")
    firefox_options.add_argument("--window-size=1920,1080")
    firefox_options.set_preference("dom.webdriver.enabled", False)  # Disable WebDriver detection
    firefox_options.set_preference("useAutomationExtension", False)
    firefox_options.set_preference("general.useragent.override", "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:138.0) Gecko/20100101 Firefox/138.0")  # Match Firefox version

    try:
        driver = webdriver.Firefox(
            service=Service(geckodriver_path),
            options=firefox_options
        )
    except Exception as e:
        raise Exception(f"Failed to start GeckoDriver: {e}. Ensure GeckoDriver v0.37.0 or later is used for Firefox 138.0.1.")

    all_comments = []

    for url in urls:
        # Resolving shortened URL
        canonical_url = resolve_youtube_url(url)
        st.write(f"Scraping comments for: {canonical_url}")
        driver.get(canonical_url)
        time.sleep(random.uniform(10, 15))  # Random initial wait to mimic human behavior

        try:
            # Scroll to ensure comment section is in view
            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(random.uniform(3, 5))

            # Waiting for comment section to load
            st.write("Waiting for comment section...")
            WebDriverWait(driver, 40).until(
                EC.presence_of_element_located((By.ID, "comments"))
            )
            st.success("Comment section container found.")

            # Check for restrictions
            try:
                comments_disabled = driver.find_element(By.CSS_SELECTOR, "#comments #message").text
                if "Comments are turned off" in comments_disabled:
                    st.warning(f"Comments are disabled for {canonical_url}")
                    continue
                if "age-restricted" in comments_disabled.lower() or "sign in" in comments_disabled.lower():
                    st.warning(f"Comments may require login or age verification for {canonical_url}")
                    continue
            except NoSuchElementException:
                pass

            # Checking if comments exist by looking for the comment count
            try:
                comment_count_elem = driver.find_element(By.CSS_SELECTOR, "#comments #count yt-formatted-string, #count .count-text yt-formatted-string")
                comment_count = comment_count_elem.text.strip()
                st.write(f"Comment count: {comment_count}")
            except NoSuchElementException:
                st.warning(f"Could not find comment count for {canonical_url}. Attempting to proceed...")
                comment_count = "Unknown"

            # Scrolling to load comments dynamically
            scroll_attempts = 0
            max_scroll_attempts = 30
            last_comment_count = 0
            while scroll_attempts < max_scroll_attempts:
                # Scroll with random increments to mimic human behavior
                driver.execute_script("window.scrollBy(0, arguments[0]);", random.randint(400, 800))
                time.sleep(random.uniform(2, 4))
                # Check for "Show more" or continuation buttons
                try:
                    show_more_button = driver.find_element(By.CSS_SELECTOR, "#comments ytd-button-renderer#more, #comments ytd-continuation-item-renderer")
                    driver.execute_script("arguments[0].scrollIntoView(true);", show_more_button)
                    show_more_button.click()
                    time.sleep(random.uniform(2, 3))
                    st.write("Clicked 'Show more' or continuation button.")
                except NoSuchElementException:
                    pass
                # Try primary selector
                comment_elements = driver.find_elements(By.CSS_SELECTOR, "ytd-comment-thread-renderer")
                current_count = len(comment_elements)
                st.write(f"Scroll attempt {scroll_attempts + 1}, found {current_count} ytd-comment-thread-renderer elements")
                if current_count >= max_comments or current_count == last_comment_count:
                    break
                last_comment_count = current_count
                scroll_attempts += 1

            # Fallback to ytd-comment-view-model
            if len(comment_elements) == 0:
                st.warning("No ytd-comment-thread-renderer elements found. Trying ytd-comment-view-model as fallback.")
                comment_elements = driver.find_elements(By.CSS_SELECTOR, "ytd-comment-view-model")
                st.write(f"Found {len(comment_elements)} ytd-comment-view-model elements.")

            # Fallback to broader selector
            if len(comment_elements) == 0:
                st.warning("No ytd-comment-view-model elements found. Trying broader selector as final fallback.")
                comment_elements = driver.find_elements(By.CSS_SELECTOR, "[id='content-text']")
                st.write(f"Found {len(comment_elements)} elements with id='content-text'.")

            # Extracting comment details
            st.write("Extracting comments...")
            st.write(f"Found {len(comment_elements)} comment elements.")
            if len(comment_elements) == 0:
                try:
                    comment_section = driver.find_element(By.ID, "comments")
                    html_snippet = comment_section.get_attribute("innerHTML")[:1000]
                    st.write(f"Comment section HTML snippet: {html_snippet}")
                except Exception as e:
                    st.error(f"Couldn’t get HTML: {e}")
                st.error("No comment elements detected. Likely selector issue or comments not loaded.")
                continue

            for i, comment_elem in enumerate(comment_elements):
                if i >= max_comments:
                    break
                try:
                    # Determine the parent element to extract details
                    try:
                        parent = comment_elem.find_element(By.XPATH, "./ancestor::ytd-comment-thread-renderer | ./ancestor::ytd-comment-view-model")
                    except NoSuchElementException:
                        parent = comment_elem

                    # Extracting commenter ID (channel ID)
                    try:
                        author_elem = parent.find_element(By.CSS_SELECTOR, "#author-text")
                        commenter_id = author_elem.get_attribute("href").split("/")[-1] if author_elem.get_attribute("href") else "unknown"
                        commenter_name = author_elem.text.strip()
                    except NoSuchElementException:
                        commenter_id = "unknown"
                        commenter_name = "unknown"

                    # Extracting comment text
                    comment_text = comment_elem.text.strip() if comment_elem.get_attribute("id") == "content-text" else comment_elem.find_element(By.CSS_SELECTOR, "#content-text").text.strip()

                    # Extracting timestamp
                    try:
                        time_elem = parent.find_element(By.CSS_SELECTOR, "span#published-time-text a, #published-time-text")
                        published_at = time_elem.text.strip()
                    except NoSuchElementException:
                        published_at = "unknown"

                    # Extracting replies
                    replies = []
                    try:
                        # Check if there are replies and a "View replies" button
                        reply_button = parent.find_elements(By.CSS_SELECTOR, "#replies ytd-button-renderer#more-replies")
                        if reply_button:
                            driver.execute_script("arguments[0].scrollIntoView(true);", reply_button[0])
                            reply_button[0].click()
                            time.sleep(random.uniform(1, 2))
                        reply_elements = parent.find_elements(By.CSS_SELECTOR, "ytd-comment-replies-renderer ytd-comment-renderer, #replies ytd-comment-view-model, #replies [id='content-text']")
                        for reply_elem in reply_elements:
                            reply_text = reply_elem.text.strip() if reply_elem.get_attribute("id") == "content-text" else reply_elem.find_element(By.CSS_SELECTOR, "#content-text").text.strip()
                            try:
                                reply_author = reply_elem.find_element(By.CSS_SELECTOR, "#author-text").text.strip()
                            except NoSuchElementException:
                                reply_author = "unknown"
                            replies.append(f"{reply_author}: {reply_text}")
                    except (NoSuchElementException, ElementClickInterceptedException):
                        replies = []

                    comment_data = {
                        "url": canonical_url,
                        "video_id": canonical_url.split("v=")[-1].split("&")[0] if "v=" in canonical_url else canonical_url.split("/")[-1],
                        "commenter_id": commenter_id,
                        "commenter_name": commenter_name,
                        "comment_text": comment_text,
                        "replies": "; ".join(replies) if replies else "",
                        "published_at": published_at,
                        "Toxic_flag": None,
                        "Toxic_type": None,
                        "lang": "unknown"
                    }
                    all_comments.append(comment_data)
                    st.write(f"Collected comment {i+1}: {comment_data['comment_text'][:50]}... (Commenter: {commenter_name}, ID: {commenter_id}, Replies: {len(replies)})")

                except NoSuchElementException as e:
                    st.error(f"Error processing comment {i+1}: Missing element - {e}")
                    continue
                except Exception as e:
                    st.error(f"Error processing comment {i+1}: {e}")
                    continue

        except TimeoutException:
            st.error(f"Timeout loading comments for {canonical_url}. Page too slow or comments unavailable.")
        except Exception as e:
            st.error(f"Error scraping {canonical_url}: {e}")

    # Saving comments to CSV
    driver.quit()
    if all_comments:
        df = pd.DataFrame(all_comments)
        df.to_csv(output_file, index=False, encoding="utf-8")
        st.success(f"Saved {len(all_comments)} comments to {output_file}")
        return df
    else:
        st.warning("No comments collected for any videos.")
        return None

def render_data_collection_ui(data_path):
    """
    Renders the Streamlit UI for YouTube comments scraping.
    """
    st.markdown("<h2>📺 YouTube Comments Scraper</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:16px;'>
    Enter YouTube video URLs to scrape comments for toxicity analysis. The scraped comments
    will be saved to the YouTube dataset and can be appended to the existing dataset.
    </div>
    """, unsafe_allow_html=True)

    # Input for YouTube URLs
    urls_input = st.text_area(
        "Enter YouTube Video URLs (one per line)",
        placeholder="https://www.youtube.com/watch?v=VIDEO_ID\nhttps://youtu.be/VIDEO_ID",
        height=150
    )

    # Input for maximum comments
    max_comments = st.number_input(
        "Maximum Comments per Video",
        min_value=10,
        max_value=1000,
        value=100,
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
            st.error("Please provide at least one YouTube URL.")
            return

        # Processing URLs
        urls = [url.strip() for url in urls_input.split("\n") if url.strip()]
        st.write(f"Found {len(urls)} URLs to scrape.")

        # Running scraper
        with st.spinner("Scraping YouTube comments... This may take a while."):
            scraped_df = scrape_youtube_comments_selenium(urls, output_file, max_comments)

        # Displaying results
        if scraped_df is not None:
            st.dataframe(scraped_df, use_container_width=True)
            st.markdown(f"**Total Comments Scraped**: {len(scraped_df)}")

            # Option to append to existing dataset
            append_to_existing = st.checkbox("Append scraped comments to existing dataset?", value=False)
            if append_to_existing:
                try:
                    existing_df = pd.read_csv(data_path) if os.path.exists(data_path) else pd.DataFrame(columns=["url", "video_id", "commenter_id", "commenter_name", "comment_text", "replies", "published_at", "Toxic_flag", "Toxic_type", "lang"])
                    updated_df = pd.concat([existing_df, scraped_df], ignore_index=True)
                    updated_df.to_csv(data_path, index=False, encoding="utf-8")
                    st.success(f"Appended {len(scraped_df)} comments to {data_path}")
                except Exception as e:
                    st.error(f"Error appending to dataset: {e}")