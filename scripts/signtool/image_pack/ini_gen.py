#!/usr/bin/python
# -*- coding: UTF-8 -*-
#----------------------------------------------------------------------------
# Purpose:
# Copyright Huawei Technologies Co., Ltd. 2010-2022. All rights reserved.
# Author: fanwenyue
#----------------------------------------------------------------------------

import xml.etree.ElementTree as ET
import hashlib
import argparse
import textwrap
import os


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent('''
                                     A tool to generate a image ini file'''))
    parser.add_argument('-in_xml', required=True, dest='inFilePath', help='The xml for get image list')
    parser.add_argument('--hash_list', required=False, help="gen cms image hash_list file", action="store_true")
    parser.add_argument('-hash_dest', required=False, dest='hash_list_path', help='hash_list file dest address')
    parser.add_argument('--hash_update', required=False, dest='new_image_name', help="update image hash to hashlist")
    parser.add_argument('-hash_list_img', required=False, dest='hash_list_img_path', help='hash_list img file path to add new hash')
    return parser.parse_args()


def validate_path(path, allow_absolute=False):
    """验证路径安全性，防止路径遍历攻击"""
    if not path:
        return None

    # 归一化路径
    normalized_path = os.path.normpath(path)

    # 检查是否包含路径遍历
    if '..' in normalized_path.split(os.sep):
        print(f"Error: Path contains '..' (path traversal): {path}")
        return None

    # 如果不允许绝对路径，检查是否为绝对路径
    if not allow_absolute and os.path.isabs(normalized_path):
        print(f"Error: Absolute path not allowed: {path}")
        return None

    return normalized_path


def cal_image_hash(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def cal_fs_image_hash(filepath, roothash):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    hash_val = sha256_hash.hexdigest() + ";dm-roothash," + roothash
    print(hash_val)
    return hash_val

def gen_ini():
    args = get_args()
    tree = ET.ElementTree(file=args.inFilePath)
    if tree.getroot().tag != 'image_info':
        print("error in input xml file")
        return -1
    if args.hash_list:
        # 验证 hash_list_path 路径安全性
        validated_path = validate_path(args.hash_list_path)
        if validated_path is None:
            print("Error: Invalid hash_list_path")
            return -1
        hash_list_path = os.path.join(validated_path, ('{}.img'.format('hash-list')))
        if (os.path.exists(hash_list_path)) :
            os.remove(hash_list_path)
        for elem in tree.iter(tag='image'):
            if elem.attrib['tag'] == 'hashlist':
                continue
            position = elem.get('position', 'after_header')
            if position == 'before_header':
                roothash = elem.get('roothash')
                hashVal = cal_fs_image_hash(elem.attrib['path'], roothash)
            else:
                hashVal = cal_image_hash(elem.attrib['path'])
            with open(hash_list_path, 'a+') as f:
                line_elem = [elem.attrib['tag'], hashVal]
                line = '{};'.format(','.join(line_elem))
                f.write(line)
    else:
        for elem in tree.iter(tag='image'):
            position = elem.get('position', 'after_header')
            if position == 'before_header':
                roothash = elem.get('roothash')
                hashVal = cal_fs_image_hash(elem.attrib['path'], roothash)
            else:
                hashVal = cal_image_hash(elem.attrib['path'])
            if hashVal == "":
                return -1
            if 'ini_name' in elem.attrib:
                file_name = os.path.join(elem.attrib['out'], f'{elem.attrib["ini_name"]}.ini')
            else:
                file_name = os.path.join(elem.attrib['out'], ('{}.ini'.format(elem.attrib['tag'])))
            # print(file_name)
            with open(file_name, 'w+') as f:
                line_elem = [elem.attrib['tag'], hashVal]
                line = '{};\n'.format(',   '.join(line_elem))
                f.write(line)
    return 0

def update_hash():
    args = get_args()
    tree = ET.ElementTree(file=args.inFilePath)
    print("update_hash")
    if tree.getroot().tag != 'image_info':
        print("error in input xml file")
        return -1
    if args.new_image_name:
        # 验证 hash_list_img_path 路径安全性（允许绝对路径）
        validated_path = validate_path(args.hash_list_img_path, allow_absolute=True)
        if validated_path is None:
            print("Error: Invalid hash_list_img_path")
            return -1
        hash_list_path = validated_path
        if (os.path.exists(hash_list_path)) :
            for elem in tree.iter(tag='image'):
                if elem.attrib['tag'] == args.new_image_name:
                    hashVal = cal_image_hash(elem.attrib['path'])
                    with open(hash_list_path, 'a+') as f:
                        line_elem = [elem.attrib['tag'], hashVal]
                        line = '{};'.format(','.join(line_elem))
                        f.write(line)
                        print('add',args.new_image_name,'hash val',hashVal,'to',hash_list_path)
        else:
            print("input hashlist file not exist")
            return 1
    return 0

def main():
    args = get_args()
    if args.new_image_name:
        update_hash()
    else:
        gen_ini()

if __name__ == '__main__':
    main()
