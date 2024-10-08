import requests
import numpy as np
from bs4 import BeautifulSoup
from bs4.element import ResultSet
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse, urljoin
from typing import Optional, Dict, List
from time import time, sleep
from dataclasses import dataclass
from re import search

@dataclass
class HTMLTag:
    html_content: str
    metadata: Dict

    def __init__(self, html_content: str = "", source: str = ""):
        self.metadata = {"source": source}
        self.html_content = html_content

@dataclass
class HTMLTagCondition:
    tag_condition: str = None
    attr_conditions: Dict = None

    def ____init__(self, tag_condition: str = None, attr_conditions: Dict = None):
        self.tag_condition = tag_condition
        self.attr_conditions = attr_conditions


class MyScraper:
    base_url: str  # The base URL of the website to scrape
    user_agent: str  # The User-Agent string for web scraping
    soup: Optional[BeautifulSoup]  # BeautifulSoup object for HTML parsing
    robots_parser: RobotFileParser  # RobotFileParser object to handle robots.txt rules
    time_last_request: float  # TBD
    wait_time_min_s: float  # TBD
    webpage_index: Dict  #
    link_search_pattern: str
    filter_conditions: List[HTMLTagCondition]
    filtered_tags: List[HTMLTag]
    current_url: str

    def __init__(self,
                 base_url: str,
                 user_agent: str = '*',
                 wait_time_min_s: float = 1.0,
                 link_search_pattern: str = r"",
                 filter_conditions: List[HTMLTagCondition] = None) -> None:

        # Initialize the base URL and user agent
        self.base_url = base_url
        self.user_agent = user_agent

        # Initialize BeautifulSoup object (set to None initially)
        self.soup = None

        # Initialize RobotFileParser object
        self.robots_parser = RobotFileParser()
        self.robots_url = urljoin(self.base_url, "/robots.txt")
        self.robots_parser.set_url(self.robots_url)
        self.robots_parser.read()

        # Initialize timer for limiting requests and min waiting time
        self.time_last_request = time()
        self.wait_time_min_s = wait_time_min_s

        # Initialize webpage_index
        self.webpage_index = {self.base_url: False}

        # Initialize html tag filter results
        self.filtered_tags = ResultSet([], [])

        # Initialize search conditions
        self.link_search_pattern = link_search_pattern
        if filter_conditions is None:
            self.filter_conditions = [HTMLTagCondition()]
        else:
            self.filter_conditions = filter_conditions

        # Initialize helper
        self.current_url = self.base_url

    def extract_links_from_website(self) -> List[str]:
        links = []
        for a_tag in self.soup.find_all("a", href=True):
            href = a_tag["href"]
            # Ensure link is absolute
            full_url = urljoin(self.base_url, href)
            # Filter internal links only
            if self.is_internal_link(full_url) and search(self.link_search_pattern, full_url) is not None:
                links.append(full_url)

        return links

    def is_internal_link(self, url):
        """Check if a link is internal (belongs to the same domain)."""
        return urlparse(url).netloc == urlparse(self.base_url).netloc

    def fetch_webpage(self, url: str):
        if not self.is_allowed(url):
            print(f"Reading prohibited: {url}")

        else:
            self.wait_for_timeout()
            response = requests.get(url)
            self.time_last_request = time()
            if response.status_code == 200:
                self.soup = BeautifulSoup(response.text, 'html.parser')
                self.current_url = url
            else:
                print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
        pass

    def prettify_webpage(self) -> str:
        return self.soup.prettify()

    def filter_webpage(self) -> None:

        for condition in self.filter_conditions:
            if condition.tag_condition and condition.attr_conditions:
                search_result = self.soup.find_all(condition.tag_condition, attrs=condition.attr_conditions)
                self.filtered_tags.extend([HTMLTag(html_content=tag.prettify(),
                                                   source=self.current_url) for tag in search_result])

            elif condition.tag_condition and not condition.attr_conditions:
                search_result = self.soup.find_all(condition.tag_condition)
                self.filtered_tags.extend([HTMLTag(html_content=tag.prettify(),
                                                   source=self.current_url) for tag in search_result])
            elif not condition.tag_condition and condition.attr_conditions:
                search_result = self.soup.find_all(attrs=condition.attr_conditions)
                self.filtered_tags.extend([HTMLTag(html_content=tag.prettify(),
                                                   source=self.current_url) for tag in search_result])
            else:
                search_result = self.soup.find(True)
                self.filtered_tags.extend([HTMLTag(html_content=tag.prettify(),
                                                   source=self.current_url) for tag in search_result])

    def wait_for_timeout(self) -> None:
        current_time = time()
        if current_time < self.time_last_request + self.wait_time_min_s:
            dt_to_timeout_end = (self.time_last_request + self.wait_time_min_s) - current_time
            sleep(dt_to_timeout_end)

    def is_allowed(self, url: str) -> bool:
        """
        Check if scraping the given URL is allowed according to robots.txt
        """
        return self.robots_parser.can_fetch(self.user_agent, url)

    def traverse_website(self):

        idx_first_unvisited = self.find_unvisited_webpage()
        while idx_first_unvisited is not None:
            current_url = list(self.webpage_index)[idx_first_unvisited]
            self.fetch_webpage(current_url)
            website_links = self.extract_links_from_website()

            # for link in website_links:
            #    if link not in self.webpage_index:
            #        self.webpage_index[link] = False
            self.webpage_index = {s: False for s in website_links} | self.webpage_index
            self.filter_webpage()
            self.webpage_index[current_url] = True
            idx_first_unvisited = self.find_unvisited_webpage()
            print(
                f"Finished Parsing {current_url}. Progress: {sum(list(self.webpage_index.values()))}/{len(self.webpage_index)}")

    def find_unvisited_webpage(self) -> Optional[int]:

        idx_first_unvisited = None
        try:
            idx_first_unvisited = list(self.webpage_index.values()).index(False)
        except:
            print("No unvisited webpage found -> Done")

        return idx_first_unvisited

    def return_filtered_tags(self, remove_duplicates: bool = True) -> List[HTMLTag]:

        if remove_duplicates:
            _, unique_idxs = np.unique([tag.html_content for tag in self.filtered_tags], return_index=True)
            return [self.filtered_tags[idx] for idx in unique_idxs]
        else:
            return self.filtered_tags

