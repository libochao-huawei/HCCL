import http.server
import json
import os
import sys
import threading

PORT = 8888

class CheckerHandler(http.server.SimpleHTTPRequestHandler):
    """自定义处理逻辑"""
    
    def do_GET(self):
        # 1. 获取服务器实例中存储的 output 路径
        output_dir = self.server.checker_output_path
        
        if not output_dir:
            self.send_error(500, "[Visualize][ERROR] Checker output directory not found")
            return

        # 接口：获取 output 文件夹下的文件列表
        if self.path == '/list_files':
            self._handle_list_files(output_dir)
        
        # 接口：获取特定文件内容 /get_file/example.json
        elif self.path.startswith('/get_file/'):
            self._handle_get_file(output_dir)
            
        else:
            # 默认返回当前目录下的静态资源（如 index.html）
            super().do_GET()

    def _handle_list_files(self, output_dir):
        """遍历 output 目录返回 JSON 列表"""
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 过滤出文件（排除文件夹）
            files = [f for f in os.listdir(output_dir) 
                    if os.path.isfile(os.path.join(output_dir, f))]
            
            self._send_json(files)
        except Exception as e:
            self.send_error(500, str(e))

    def _handle_get_file(self, output_dir):
        """安全地读取文件内容"""
        # 提取文件名并防止路径穿越 (取最后一部分)
        filename = os.path.basename(self.path)
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath) and os.path.isfile(filepath):
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            with open(filepath, 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404, "File Not Found")

    def _send_json(self, data):
        """发送 JSON 响应的辅助方法"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))


class DumperServer:
    def __init__(self):
        self.httpd = None
        self.keep_running = True
        # 启动时确定 checker/output 路径
        self.checker_output_path = self._setup_checker_path()

    def _setup_checker_path(self):
        """查找到 checker 插件并定位其 output 目录"""
        current_dir = os.path.abspath(os.path.dirname(__file__))
        plugin_root = None

        # 向上追溯 3 层寻找 plugin
        tmp_dir = current_dir
        for _ in range(4):
            potential = os.path.join(tmp_dir, "plugin")
            if os.path.isdir(potential):
                plugin_root = potential
                break
            parent = os.path.dirname(tmp_dir)
            if parent == tmp_dir: break
            tmp_dir = parent

        if not plugin_root:
            print("[Visualize][Error] 'plugin' directory not found.")
            return None

        # 遍历查找 name 为 checker 的 manifest
        for root, dirs, files in os.walk(plugin_root):
            # 限制深度：os.walk 深度控制
            depth = root[len(plugin_root):].count(os.sep)
            if depth > 2: continue 

            if "manifest.json" in files:
                m_path = os.path.join(root, "manifest.json")
                try:
                    with open(m_path, 'r') as f:
                        if json.load(f).get("name") == "checker":
                            output_path = os.path.join(root, "output")
                            if not os.path.exists(output_path):
                                os.makedirs(output_path)
                            return output_path
                except: continue
        return None

    def run_server(self):
        server_address = ('', PORT)
        # 将 output 路径挂载到 server 对象上，方便 Handler 访问
        self.httpd = http.server.HTTPServer(server_address, CheckerHandler)
        self.httpd.checker_output_path = self.checker_output_path
        
        print(f"[Visualize] Server running on port {PORT}, http://localhost:{PORT}")
        print(f"[Visualize] Serving files from: {self.checker_output_path}")
        self.httpd.serve_forever()

    def start(self):
        # 开启后台线程运行 HTTP Server
        thread = threading.Thread(target=self.run_server, daemon=True)
        thread.start()

        # 主线程监听标准输入
        try:
            while self.keep_running:
                line = sys.stdin.readline()
                if not line: break
                
                cmd = json.loads(line)
                if cmd.get("action") == "stop":
                    print("[Visualize] Received stop command.")
                    self.keep_running = False
                    if self.httpd: self.httpd.shutdown()
                elif cmd.get("action") == "status":
                    payload = cmd.get("payload")
                    if isinstance(payload, dict):
                        data_id = payload.get("data_id")
                        print(f"[Visualize][INFO] You can find output files starting with: {data_id}")
                        print(f"[Visualize][INFO] Server running: http://localhost:{PORT}")
        except Exception as e:
            print(f"[Visualize][ERROR] Main loop error: {e}")
        finally:
            print("[Visualize][INFO] Process exiting.")

if __name__ == "__main__":
    DumperServer().start()