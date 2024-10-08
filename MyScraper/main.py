from MyScraper import MyScraper, HTMLTagCondition


def main(scrape_url_base: str):
    # Create hmtl filter conditions
    filter_conditions = []
    cond1 = HTMLTagCondition(tag_condition="div", attr_conditions={"class": "quote"})
    filter_conditions.append(cond1)
    # Create link
    link_search_pattern = r"https://quotes.toscrape.com/page/\d/"

    my_scraper = MyScraper(base_url,
                           wait_time_min_s=0.5,
                           filter_conditions=filter_conditions,
                           link_search_pattern=link_search_pattern)

    my_scraper.traverse_website()
    filtered_tags = my_scraper.return_filtered_tags()

    # my_scraper.fetch_webpage(base_url)
    # result = my_scraper.query_soup_tag("span")
    # result = my_scraper.query_soup_attr(attr_name="class", attr_value="text")
    # filter_conditions = []
    # cond1 = HTMLTagCondition(tag_condition="small")
    # cond2 = HTMLTagCondition(tag_condition="span", attr_conditions={"class": "text"})
    # filter_conditions.append(cond1)
    # filter_conditions.append(cond2)
    # cond1 = HTMLTagCondition(tag_condition="div", attr_conditions={"class": "quote"})
    # filter_conditions.append(cond1)
    # filtered_tags = my_scraper.filter_webpage(filter_conditions)
    # links = my_scraper.extract_links_from_website()

    pass


if __name__ == '__main__':
    # base_url = "https://www.scrapethissite.com/"
    base_url = "https://quotes.toscrape.com/"
    main(base_url)
