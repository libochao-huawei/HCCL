#!/usr/bin/env python
#-*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Purpose:
# Copyright Huawei Technologies Co., Ltd. 2010-2025. All rights reserved.
#----------------------------------------------------------------------------
import argparse
import textwrap
import shutil
import os
import struct

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent(''' A tool to recover raw img'''))
    parser.add_argument('-img', required=True, dest='img',
                        help='INPUT: img with header, Parse N/E from it')
    parser.add_argument('-raw', required=False, dest='raw',
                        help='INPUT: recovered raw img')
    parser.add_argument('--rcvr', help="recover the raw img without header",
                        action="store_true")

    return parser.parse_args()


def __write_raw_img(raw, img, code_len, max_copy_size):
    raw.seek(0)
    img.seek(0x2100)
    # 限制拷贝长度：取min(声明长度-0x100, 剩余文件大小)
    rsv_len = min(code_len - 0x100, max_copy_size)
    while rsv_len > 0:
        chunk_size = 4096 if rsv_len > 4096 else rsv_len
        read_buf = img.read(chunk_size)
        if not read_buf:  # 遇到 EOF 立即 break
            break
        raw.write(read_buf)
        rsv_len -= len(read_buf)  # 按实际读取量递减


def check_image_headered(file_path):
    with open(file_path, 'rb') as f:
        # 读取前4个字节
        data = f.read(4)
        if len(data) < 4:
            return False
        # 将字节转换为小端序的32位整数
        word = int.from_bytes(data, byteorder='little')
        return word == 0x55aa55aa


def main():
    args = get_args()
    if not args.rcvr:
        raise Exception("No operation found, do nothing")

    tmp_file = args.raw + '.tmp' # 中间暂存文件
    if check_image_headered(args.img) == False: # 无头场景，直接将原镜像拷到tmp
        shutil.copyfile(args.img, tmp_file)
    else:
        with open(args.img, 'rb') as img: # 有头场景，将原镜像实际镜像长度拷出到tmp
            img.seek(0x478)
            code_len = struct.unpack('<I', img.read(4))[0]
            # 检验二进制头字段：code_len必须合法范围且与文件大小一致
            if code_len < 0x100:
                raise ValueError(f"Invalid code_len: (0x{code_len:x}), must be >= 0x100")
            file_size = img.seek(0, os.SEEK_END)
            if code_len > file_size:
                raise ValueError(f"Invalid code_len: (0x{code_len:x}), must be <= file size: (0x{file_size:x})")
            max_copy_size = file_size - 0x2100
            with open(tmp_file, 'wb+') as raw:
                __write_raw_img(raw, img, code_len, max_copy_size)
    shutil.copyfile(tmp_file, args.raw)
    if os.path.exists(tmp_file):
        os.remove(tmp_file)


if __name__ == '__main__':
    main()


