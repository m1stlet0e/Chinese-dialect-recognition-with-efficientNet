# 方言识别 REST API

这是一个基于 Flask 的 REST API 服务,提供中国方言识别功能。

## 快速开始

### 1. 安装依赖

```bash
cd API
pip install -r requirements.txt
```

### 2. 启动服务

```bash
python api_server.py
```

服务将在 `http://localhost:5000` 启动

### 3. 测试API

```bash
# 使用测试客户端
python test_client.py <音频文件.wav>
```

## API 接口文档

### 基础信息

- **服务地址**: `http://localhost:5000`
- **请求格式**: `multipart/form-data` (文件上传)
- **响应格式**: `application/json`

---

### 1. 健康检查

检查服务是否正常运行

**请求**
```
GET /api/health
```

**响应**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

---

### 2. 获取支持的方言列表

**请求**
```
GET /api/dialects
```

**响应**
```json
{
  "dialects": [
    "长沙话",
    "河北话",
    "合肥话",
    "客家话",
    "南昌话",
    "宁夏话",
    "陕西话",
    "上海话",
    "四川话"
  ],
  "count": 9
}
```

---

### 3. 预测单个音频文件 (主要接口)

**请求**
```
POST /api/predict
Content-Type: multipart/form-data

file: <音频文件> (WAV或MP3格式)
```

**响应 - 成功**
```json
{
  "success": true,
  "dialect": "客家话",
  "confidence": 0.8555,
  "all_probabilities": {
    "客家话": 0.8555,
    "长沙话": 0.0937,
    "南昌话": 0.0234,
    "河北话": 0.0112,
    "陕西话": 0.0086,
    "宁夏话": 0.0057,
    "四川话": 0.0019,
    "上海话": 0.0000,
    "合肥话": 0.0000
  }
}
```

**响应 - 失败**
```json
{
  "success": false,
  "error": "错误信息"
}
```

---

### 4. 批量预测多个音频文件

**请求**
```
POST /api/predict_batch
Content-Type: multipart/form-data

files: <音频文件1>
files: <音频文件2>
files: <音频文件3>
...
```

**响应**
```json
{
  "success": true,
  "total": 3,
  "results": [
    {
      "filename": "test1.wav",
      "success": true,
      "dialect": "客家话",
      "confidence": 0.8555,
      "all_probabilities": {...}
    },
    {
      "filename": "test2.wav",
      "success": true,
      "dialect": "四川话",
      "confidence": 0.9123,
      "all_probabilities": {...}
    }
  ]
}
```

---

## 使用示例

### Python (requests)

```python
import requests

# 预测单个文件
with open('audio.wav', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/api/predict', files=files)
    result = response.json()
    
    if result['success']:
        print(f"识别结果: {result['dialect']}")
        print(f"置信度: {result['confidence']:.2%}")
```

### cURL

```bash
# 健康检查
curl http://localhost:5000/api/health

# 获取方言列表
curl http://localhost:5000/api/dialects

# 预测音频文件
curl -X POST -F "file=@audio.wav" http://localhost:5000/api/predict

# 批量预测
curl -X POST \
  -F "files=@audio1.wav" \
  -F "files=@audio2.wav" \
  http://localhost:5000/api/predict_batch
```

### JavaScript (fetch)

```javascript
// 预测音频文件
const formData = new FormData();
formData.append('file', audioFile);

fetch('http://localhost:5000/api/predict', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  if (data.success) {
    console.log('识别结果:', data.dialect);
    console.log('置信度:', (data.confidence * 100).toFixed(2) + '%');
  }
});
```

### Java (OkHttp)

```java
OkHttpClient client = new OkHttpClient();

RequestBody requestBody = new MultipartBody.Builder()
    .setType(MultipartBody.FORM)
    .addFormDataPart("file", "audio.wav",
        RequestBody.create(new File("audio.wav"), MediaType.parse("audio/wav")))
    .build();

Request request = new Request.Builder()
    .url("http://localhost:5000/api/predict")
    .post(requestBody)
    .build();

Response response = client.newCall(request).execute();
String result = response.body().string();
```

---

## 错误码说明

| HTTP状态码 | 说明 |
|-----------|------|
| 200 | 请求成功 |
| 400 | 请求参数错误(如文件格式不支持) |
| 500 | 服务器内部错误 |

---

## 配置说明

可在 `api_server.py` 中修改以下配置:

```python
# 端口号
app.run(host='0.0.0.0', port=5000)

# 最大文件大小 (默认16MB)
MAX_CONTENT_LENGTH = 16 * 1024 * 1024

# 允许的文件格式
ALLOWED_EXTENSIONS = {'wav', 'mp3'}
```

---

## 性能说明

- **单次请求**: 2-5秒 (CPU模式)
- **并发支持**: 支持多客户端同时请求
- **建议**: 生产环境使用 Gunicorn 或 uWSGI

### 生产部署 (使用 Gunicorn)

```bash
# 安装 gunicorn
pip install gunicorn

# 启动服务 (4个worker进程)
gunicorn -w 4 -b 0.0.0.0:5000 api_server:app
```

---

## 注意事项

1. **音频格式**: 
   - 推荐使用WAV格式
   - MP3也支持,但需要ffmpeg
   - 音频时长: 5-10秒最佳

2. **文件大小**: 
   - 单个文件不超过16MB
   - 批量请求建议不超过10个文件

3. **跨域**: 
   - API已启用CORS支持
   - 可从任何域名调用

4. **安全**: 
   - 默认无认证,仅供内网使用
   - 生产环境建议添加API Key认证

---

## 完整示例

详细的调用示例请参考 `test_client.py` 文件。

运行测试:
```bash
# 测试单个文件
python test_client.py audio.wav

# 测试多个文件
python test_client.py audio1.wav audio2.wav audio3.wav
```

---

## 常见问题

### Q: 启动服务失败?
A: 确保安装了所有依赖,并且模型文件 `../GUI/weight/model-29.pth` 存在

### Q: 上传文件后返回500错误?
A: 检查音频文件格式是否正确,查看服务端日志获取详细错误信息

### Q: 如何提高识别速度?
A: 使用GPU设备,或使用Gunicorn启动多进程服务

### Q: 支持实时音频流吗?
A: 当前版本仅支持文件上传,不支持音频流

---

## 技术栈

- **Web框架**: Flask 2.3+
- **深度学习**: PyTorch, torchvision
- **模型**: EfficientNet-B0
- **音频处理**: SciPy, NumPy
- **图像处理**: Pillow

---

## 许可证

本项目仅供学习和研究使用。


