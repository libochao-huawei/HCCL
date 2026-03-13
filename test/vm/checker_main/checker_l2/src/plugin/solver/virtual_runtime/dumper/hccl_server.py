import http.server
import socketserver
import json
import os

PORT = 8000
# 设置存放 HCCL dump 文件的目录
DATA_DIRECTORY = "./output" 


class HCCLHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        """处理 GET 请求"""
        
        # 接口：获取 JSON 文件列表
        if self.path == '/list_files':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # 确保数据目录存在
            if not os.path.exists(DATA_DIRECTORY):
                os.makedirs(DATA_DIRECTORY)
                
            # 获取目录下所有 JSON 文件
            files = [f for f in os.listdir(DATA_DIRECTORY) if f.endswith('.json')]
            self.wfile.write(json.dumps(files).encode())
        
        # 接口：获取特定文件内容
        elif self.path.startswith('/get_file/'):
            filename = self.path.split('/')[-1]
            filepath = os.path.join(DATA_DIRECTORY, filename)
            
            if os.path.exists(filepath):
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                # 返回文件内容
                with open(filepath, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "File Not Found")
        
        else:
            # 默认返回 HTML 页面（如果你将 HTML 命名为 index.html）
            super().do_GET()


def run(server_class=http.server.HTTPServer, handler_class=HCCLHandler):
    """启动 HTTP 服务器"""
    server_address = ('', PORT)
    httpd = server_class(server_address, handler_class)
    print(f"Server running at http://localhost:{PORT}")
    httpd.serve_forever()


if __name__ == "__main__":
    run()
