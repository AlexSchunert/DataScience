import requests
from bs4 import BeautifulSoup
from bs4.element import ResultSet
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse, urljoin
from typing import Optional
from time import time, sleep


class MyScraper:
    base_url: str  # The base URL of the website to scrape
    user_agent: str  # The User-Agent string for web scraping
    soup: Optional[BeautifulSoup]  # BeautifulSoup object for HTML parsing
    robots_parser: RobotFileParser  # RobotFileParser object to handle robots.txt rules
    time_last_request: float # TBD
    wait_time_min_s: float # TBD

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

    def fetch_website_content(self, url: str):
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

    def query_soup_tag(self, tag_name: str):
        return self.soup.find_all(tag_name)

    def query_soup_attr(self, attr_name: str, attr_value: str) -> ResultSet:
        return self.soup.find_all(attrs={attr_name: attr_value})


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
