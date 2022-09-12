#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
爬取百度图片
@author : Chen Gavin
@data   : 2022/9/11 23:10
"""
import json
import math
import os
import re
import sys

import requests


headers_objUrl = {
    'X-Requested-With': 'XMLHttpRequest',
    'Accept': '*/*',
    'Accept-Encoding': 'gzip, deflate, br',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/96.0.4664.110 Safari/537.36 Edg/96.0.1054.62',
}


def request_url(pn):
    url = 'https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ie=utf-8&oe=utf-8'
    headers = {
        'X-Requested-With': 'XMLHttpRequest',
        'Accept': '*/*',
        'Accept-Encoding': 'gzip, deflate, br',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/96.0.4664.110 Safari/537.36 Edg/96.0.1054.62',
        'Host': 'image.baidu.com',
    }
    params = {
        'logid': '10616783841481369264',
        'word': words,
        'queryWord': words,
        'width': width,
        'height': height,
        'z': z,
        'pn': pn,
        'rn': size
    }
    return requests.get(url, params=params, headers=headers)


def baidu_decrypt(url):  # 因为要获取的是objURL所以这里找了一个百度解密
    res = ''
    c = ['_z2C$q', '_z&e3B', 'AzdH3F']
    d = {'w': 'a', 'k': 'b', 'v': 'c', '1': 'd', 'j': 'e', 'u': 'f', '2': 'g', 'i': 'h', 't': 'i', '3': 'j', 'h': 'k',
         's': 'l', '4': 'm', 'g': 'n', '5': 'o', 'r': 'p', 'q': 'q', '6': 'r', 'f': 's', 'p': 't', '7': 'u', 'e': 'v',
         'o': 'w', '8': '1', 'd': '2', 'n': '3', '9': '4', 'c': '5', 'm': '6', '0': '7', 'b': '8', 'l': '9', 'a': '0',
         '_z2C$q': ':', '_z&e3B': '.', 'AzdH3F': '/'}
    if url is None or 'http' in url:
        return url
    else:
        j = url
        for m in c:
            j = j.replace(m, d[m])
        for char in j:
            if re.match(r'^[a-w\d]+$', char):
                char = d[char]
            res = res + char
        return res


def parse_result(result, img_path, total, actual):

    result_text = result.text.replace('\\\'', '\\\"')
    result_json = json.loads(result_text)
    if result_json['data'] is None:
        return
    for data in result_json['data']:
        if data is None or {} == data:
            continue
        total += 1
        if data['objURL'] is None:
            continue
        obj_url = baidu_decrypt(data['objURL'])
        rs = requests.get(obj_url, headers=headers_objUrl)
        # 忽略大小低于此值的图片；有些是请求不到，返回的数据是错误的
        if len(rs.content) < 100:
            continue
        if data["fromPageTitleEnc"] is None:
            title = 'title'
        else:
            title = re.sub(r'\W+', '', str(data["fromPageTitleEnc"]))
        if data["type"] is None:
            suffix = 'png'
        else:
            suffix = data["type"]
        filename = f'{actual}-{title}.{suffix}'
        with open(f'{img_path}{filename}', 'wb') as f:
            f.write(rs.content)
        actual += 1
    return total, actual


def hand_result(pn, img_path, total_num, actual_num):
    result = request_url(pn)
    total, actual = parse_result(result, img_path, total_num, actual_num)
    if total == total_num:
        print('已经没有更多满足条件的图片了，程序到此结束')
        sys.exit()
    actual_num = actual
    total_num = total
    print(f'本次目标数量{target_num}, 目前共爬取 {total_num} 张；已下载成功 {actual_num} 张；下载失败共 {total_num-actual_num} 张')
    return total, actual


def run():
    actual_num = 0
    total_num = 0
    page = math.ceil(target_num / size)
    img_path = base_path + words + '/'
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    for i in range(1, page + 1):
        total_num, actual_num = hand_result(i * size, img_path, total_num, actual_num)
    next_page = page + 1
    while actual_num < target_num:
        total_num, actual_num = hand_result(next_page * size, img_path, total_num, actual_num)
        next_page += 1


if __name__ == '__main__':
    # 1、增加重试连接次数
    requests.DEFAULT_RETRIES = 5
    s = requests.session()
    # 2、关闭多余的连接
    s.keep_alive = False
    # 每次请求图片数量
    size = 100
    # 目标数量
    target_num = 1000
    base_path = 'D:/桌面/relatedFile/'
    words = '狗子'
    # 百度定义的图片大小范围取值 1~9，值越大图片越大; 0-表示任意
    z = 0
    # 宽度
    width = None
    # 高度
    height = None
    run()