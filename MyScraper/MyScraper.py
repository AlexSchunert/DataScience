import requests
from bs4 import BeautifulSoup
from bs4.element import ResultSet
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse, urljoin
from typing import Optional, Dict, List
from time import time, sleep
from dataclasses import dataclass


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

    def __init__(self, base_url: str, user_agent: str = '*', wait_time_min_s: float = 1.0) -> None:

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
        pass

    def extract_links(self):
        links = []
        for a_tag in self.soup.find_all("a", href=True):
            href = a_tag["href"]
            # Ensure link is absolute
            full_url = urljoin(self.base_url, href)
            # Filter internal links only
            if self.is_internal_link(full_url):
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
            else:
                print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
        pass

    def prettify_webpage(self) -> str:
        return self.soup.prettify()

    def filter_webpage(self, filter_conditions: List[HTMLTagCondition]) -> ResultSet:

        result = ResultSet([], [])
        for condition in filter_conditions:
            if condition.tag_condition and condition.attr_conditions:
                result += self.soup.find_all(condition.tag_condition, attrs=condition.attr_conditions)
            elif condition.tag_condition and not condition.attr_conditions:
                result += self.soup.find_all(condition.tag_condition)
            elif not condition.tag_condition and condition.attr_conditions:
                result += self.soup.find_all(attrs=condition.attr_conditions)
            else:
                print("Invalid tag and attribute condition")
        return result

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
