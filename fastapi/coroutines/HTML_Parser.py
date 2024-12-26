
from html.parser import HTMLParser


class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(f"Start tag: {tag}")
        for attr in attrs:
            print(f"     attr: {attr}")

    def handle_endtag(self, tag):
        print(f"End tag  : {tag}")

    def handle_data(self, data):
        print(f"Data     : {data}")

    def handle_comment(self, data):
        print(f"Comment  : {data}")

    def handle_entityref(self, name):
        print(f"Named ent: {name}")

    def handle_charref(self, name):
        print(f"Num ent  : {name}")

# 创建解析器实例
parser = MyHTMLParser()

# 解析 HTML 文本
html_text = """
<html>
  <head><title>Test</title></head>
  <body>
    <h1>Parse me!</h1>
    <!-- This is a comment -->
    &copy; 2023
  </body>
</html>
"""
parser.feed(html_text)
print(parser.found_links)
