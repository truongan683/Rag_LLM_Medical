# modules/google_search_tamanh.py

import requests
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
import logging
from config import SECTION_CATEGORIES
from modules.chunk.section_classifier import determine_section_id_from_title

class GoogleSearchTamAnh:
    def __init__(self, api_key, cse_id):
        self.api_key = api_key
        self.cse_id = cse_id

    def search(self, query, num_results=1):
        logging.info(f"Google Search Query: {query} (top {num_results})")
        try:
            service = build("customsearch", "v1", developerKey=self.api_key)
            res = service.cse().list(q=query, cx=self.cse_id, num=num_results).execute()
            results = []
            if 'items' in res:
                for item in res['items']:
                    results.append({
                        "title": item.get('title', ''),
                        "link": item.get('link', '')
                    })
                    logging.info(f"Result: {item.get('title', '')} | {item.get('link', '')}")
            else:
                logging.warning(f"Không có kết quả cho truy vấn: {query}")
            return results
        except Exception as e:
            logging.error(f"Lỗi khi search Google API với query: {query} - {e}")
            return []


    def crawl_sections_by_h2(self, link, section_ids=None, main_obj=None):
        logging.info(f"Crawling (theo h2 section) link: {link}")
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            response = requests.get(link, headers=headers, timeout=15)
            if response.status_code != 200:
                logging.error(f"Lỗi tải link {link} - Status code: {response.status_code}")
                return []
            soup = BeautifulSoup(response.text, "html.parser")
            main_content = soup.find("div", class_="col-xs-12 col-sm-12 col-md-12 col-lg-12")
            if not main_content:
                logging.warning(f"Không tìm thấy thẻ nội dung chính ở {link}")
                return []
            h1 = main_content.find("h1")
            title = h1.text.strip() if h1 else ""
            context_blocks = []
            all_tags = list(main_content.find_all(["h2", "h3", "p", "ul", "ol"]))
            i = 0
            while i < len(all_tags):
                tag = all_tags[i]
                if tag.name == "h2":
                    section_title = tag.text.strip()
                    section_key = determine_section_id_from_title(section_title)
                    match_section = section_ids and section_key in section_ids
                    match_mainobj = main_obj and main_obj.lower() in section_title.lower()
                    if match_section or match_mainobj:
                        content = ""
                        j = i + 1
                        while j < len(all_tags) and all_tags[j].name != "h2":
                            sub_tag = all_tags[j]
                            if sub_tag.name == "p":
                                text = " ".join(sub_tag.text.split())
                                if "HỆ THỐNG BỆNH VIỆN ĐA KHOA TÂM ANH" in text:
                                    break
                                content += text + "\n"
                            elif sub_tag.name == "h3":
                                content += f"\nh3 {sub_tag.text.strip()}\n"
                            elif sub_tag.name in ["ul", "ol"]:
                                for li in sub_tag.find_all("li"):
                                    content += f"- {li.text.strip()}\n"
                            j += 1
                        block = {
                            "section": section_key if match_section else "",
                            "main_obj": main_obj if match_mainobj else "",
                            "section_title": section_title,
                            "content": content.strip(),
                            "link": link
                        }
                        context_blocks.append(block)
                        i = j  # chỉ gán i = j nếu đã có j
                    else:
                        i += 1  # KHÔNG vào nhánh if, thì chỉ tăng 1
                else:
                    i += 1

            logging.info(f"Đã lấy {len(context_blocks)} section phù hợp từ link {link}")
            return context_blocks
        except Exception as e:
            logging.error(f"Lỗi khi crawl section_by_h2 {link}: {e}")
            return []

    def get_context(self, prompt, section_ids=None, main_obj=None):
        logging.info(f"Bắt đầu get_context với prompt: {prompt}")
        results = self.search(prompt, num_results=1)
        if not results:
            logging.warning(f"Không tìm thấy kết quả Google cho: {prompt}")
            return None
        link = results[0]['link']
        context_blocks = self.crawl_sections_by_h2(link, section_ids, main_obj)
        if not context_blocks:
            logging.warning(f"Không crawl được section nào phù hợp cho: {link}")
            return None
        # Build context rõ ràng theo format yêu cầu
        context_list = []
        for block in context_blocks:
            block_str = f"{block['link']} section: {block['section']} ({block['section_title']}) {block['content']}"
            if block["main_obj"]:
                block_str += f" | main_obj: {block['main_obj']} {block['content']}"
            context_list.append(block_str)
        context = "\n\n".join(context_list)
        logging.info(f"Context build thành công ({len(context_blocks)} section, dài {len(context)} ký tự)")
        return context
