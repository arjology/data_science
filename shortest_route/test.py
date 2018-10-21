#!/usr/local/bin/python3

# myapp.py
import logging

def main():
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level='INFO')
    logger = logging.getLogger("test")
    logger.info('Started')
    logger.info('Finished')

if __name__ == '__main__':
    main()
