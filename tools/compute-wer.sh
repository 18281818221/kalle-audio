#!/bin/bash

#!/bin/bash

# 检查是否提供了足够的参数
if [ $# -lt 3 ]; then
  echo "Usage: $0 <language> <number> <path>"
  exit 1
fi

# 接收路径参数
path="$3"

# 接收语言参数
lang_param="$1"

# 接收显卡参数
cuda_param="$2"

# 打印接收到的路径和参数
echo "The provided path is: $path , Language parameter: $lang_param , Number parameter: $cuda_param"

# 执行第一个 Python 命令
python asr_test.py $lang_param $cuda_param $path

# 执行原来的 Python 命令
python compute-wer.py --char=1 --v=1 $path/aaa_gt.txt $path/aaa_asr.txt > $path/000000000_wer.txt

# # 检查是否提供了路径参数
# if [ -z "$1" ]; then
#   echo "Usage: $0 <path>"
#   exit 1
# fi

# # 接收路径参数
# path="$1"

# # 打印接收到的路径
# echo "The provided path is: $path"
# python compute-wer.py --char=1 --v=1 $path/aaa_gt.txt $path/aaa_asr.txt > $path/000000000_wer.txt

