# ch15_28.py
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s : %(message)s')
logging.debug('程式開始')
for i in range(5):
    logging.debug('目前索引 %s ' % i)
logging.debug('程式結束')


