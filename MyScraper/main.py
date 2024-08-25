from MyScraper import MyScraper


def main(scrape_url_base: str):

    my_scraper = MyScraper(base_url)
    my_scraper.fetch_website_content(base_url)
    result = my_scraper.query_soup_tag("span")
    result = my_scraper.query_soup_attr(attr_name="class", attr_value="text")

    pass




if __name__ == '__main__':
    #base_url = "https://www.scrapethissite.com/"
    base_url = "https://quotes.toscrape.com/"
    main(base_url)
