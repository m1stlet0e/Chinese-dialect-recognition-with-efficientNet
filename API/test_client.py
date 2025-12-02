"""
API 客户端测试脚本
演示如何调用方言识别API
"""

import requests
import sys
import json

# API 服务地址
API_URL = "http://localhost:8000"


def test_health():
    """测试健康检查接口"""
    print("\n" + "="*50)
    print("测试: 健康检查接口")
    print("="*50)
    
    response = requests.get(f"{API_URL}/api/health")
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")


def test_dialects():
    """测试方言列表接口"""
    print("\n" + "="*50)
    print("测试: 获取支持的方言列表")
    print("="*50)
    
    response = requests.get(f"{API_URL}/api/dialects")
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")


def test_predict(audio_file):
    """测试预测接口"""
    print("\n" + "="*50)
    print(f"测试: 预测音频文件 - {audio_file}")
    print("="*50)
    
    with open(audio_file, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{API_URL}/api/predict", files=files)
    
    print(f"状态码: {response.status_code}")
    result = response.json()
    
    if result.get('success'):
        print(f"\n识别结果: {result['dialect']}")
        print(f"置信度: {result['confidence']:.2%}")
        print(f"\n所有方言概率:")
        
        # 按概率排序
        sorted_probs = sorted(result['all_probabilities'].items(), 
                             key=lambda x: x[1], reverse=True)
        for dialect, prob in sorted_probs:
            print(f"  {dialect}: {prob:.2%}")
    else:
        print(f"错误: {result.get('error')}")


def test_predict_batch(audio_files):
    """测试批量预测接口"""
    print("\n" + "="*50)
    print(f"测试: 批量预测 {len(audio_files)} 个音频文件")
    print("="*50)
    
    files = [('files', open(f, 'rb')) for f in audio_files]
    response = requests.post(f"{API_URL}/api/predict_batch", files=files)
    
    # 关闭文件
    for _, f in files:
        f.close()
    
    print(f"状态码: {response.status_code}")
    result = response.json()
    
    if result.get('success'):
        print(f"\n处理了 {result['total']} 个文件:")
        for item in result['results']:
            print(f"\n文件: {item['filename']}")
            if item.get('success'):
                print(f"  识别结果: {item['dialect']}")
                print(f"  置信度: {item['confidence']:.2%}")
            else:
                print(f"  错误: {item.get('error')}")
    else:
        print(f"错误: {result.get('error')}")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("方言识别 API 客户端测试工具")
    print("="*60)
    
    # 测试健康检查
    try:
        test_health()
    except Exception as e:
        print(f"健康检查失败: {e}")
        print("请确保API服务已启动: python api_server.py")
        return
    
    # 测试方言列表
    test_dialects()
    
    # 测试预测
    if len(sys.argv) > 1:
        audio_files = sys.argv[1:]
        
        if len(audio_files) == 1:
            test_predict(audio_files[0])
        else:
            test_predict_batch(audio_files)
    else:
        print("\n" + "="*60)
        print("使用方法:")
        print("  单个文件: python test_client.py <音频文件.wav>")
        print("  多个文件: python test_client.py file1.wav file2.wav ...")
        print("="*60)


if __name__ == '__main__':
    main()

