import queue
import threading
import time
import logging
from typing import List, Dict, Any
import torch
from torch.utils.data import IterableDataset
from tqdm.auto import tqdm

class DataPrefetchPool:
    """
    数据预加载池，使用生产者-消费者模式实现数据的异步加载
    """
    def __init__(self, dataset, max_size=100, num_workers=2, collate_fn=None):
        """
        初始化数据预加载池
        
        Args:
            dataset: 数据集对象
            max_size: 数据池最大容量
            num_workers: 数据加载线程数
            collate_fn: 数据整理函数
        """
        self.dataset = dataset
        self.data_queue = queue.Queue(maxsize=max_size)
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.stop_event = threading.Event()
        self.workers = []
        self.dataset_size = len(dataset)
        self.current_index = 0
        self.lock = threading.Lock()
        
    def _worker_fn(self, worker_id):
        """数据加载工作线程函数"""
        while not self.stop_event.is_set():
            if self.data_queue.qsize() >= self.data_queue.maxsize * 0.9:
                # 如果队列接近满，休息一下
                time.sleep(0.1)
                continue
                
            with self.lock:
                if self.current_index >= self.dataset_size:
                    self.current_index = 0  # 重置索引，实现数据循环
                    
                idx = self.current_index
                self.current_index += 1
                
            if idx < self.dataset_size:
                try:
                    item = self.dataset[idx]
                    self.data_queue.put(item, block=True, timeout=5)
                except Exception as e:
                    logging.error(f"Worker {worker_id} error loading data at index {idx}: {e}")
                    time.sleep(0.1)  # 出错后稍微等待
    
    def start(self):
        """启动数据加载线程"""
        logging.info(f"Starting data prefetch pool with {self.num_workers} workers")
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_fn, 
                args=(i,),
                daemon=True,
                name=f"DataLoader-{i}"
            )
            worker.start()
            self.workers.append(worker)
    
    def get_batch(self, batch_size):
        """
        获取一个批次的数据
        
        Args:
            batch_size: 批次大小
            
        Returns:
            一个批次的数据
        """
        batch = []
        for _ in range(batch_size):
            try:
                item = self.data_queue.get(block=True, timeout=5)
                batch.append(item)
                self.data_queue.task_done()
            except queue.Empty:
                if len(batch) > 0:
                    break
                else:
                    logging.warning("Timeout waiting for data")
        
        if self.collate_fn is not None and len(batch) > 0:
            return self.collate_fn(batch)
        return batch
    
    def stop(self):
        """停止数据加载"""
        self.stop_event.set()
        for worker in self.workers:
            worker.join(timeout=2)
        logging.info("Data prefetch pool stopped")
            
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        
    def get_status(self):
        """获取数据池状态信息"""
        return {
            "queue_size": self.data_queue.qsize(),
            "max_size": self.data_queue.maxsize,
            "workers": len(self.workers),
            "current_index": self.current_index,
            "dataset_size": self.dataset_size
        }



class PrefetchDataIterator:
    """
    预加载数据迭代器，封装DataPrefetchPool提供类似DataLoader的接口
    """
    def __init__(self, dataset, batch_size=1, max_pool_size=100, num_workers=2, collate_fn=None):
        """
        初始化预加载数据迭代器
        
        Args:
            dataset: 数据集对象
            batch_size: 批次大小
            max_pool_size: 数据池最大容量
            num_workers: 数据加载线程数
            collate_fn: 数据整理函数
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.data_pool = DataPrefetchPool(
            dataset=dataset,
            max_size=max_pool_size,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
        self.steps_per_epoch = len(dataset) // batch_size
        self.data_pool.start()
        
    def __iter__(self):
        return self
        
    def __next__(self):
        batch = self.data_pool.get_batch(self.batch_size)
        if not batch:
            raise StopIteration
        return batch
    
    def __len__(self):
        return self.steps_per_epoch
    
    def __del__(self):
        self.data_pool.stop()
        
class DynamicPrefetchBatchIterator:
    """
    动态批次大小的预加载数据迭代器，与DynamicBatchGenerator结合使用
    """
    def __init__(self, dataset, batch_generator, max_pool_size=100, num_workers=2):
        """
        初始化动态批次预加载数据迭代器
        
        Args:
            dataset: 数据集对象
            batch_generator: 动态批次生成器对象
            max_pool_size: 数据池最大容量
            num_workers: 数据加载线程数
        """
        self.dataset = dataset
        self.batch_generator = batch_generator
        self.data_pool = DataPrefetchPool(
            dataset=dataset,
            max_size=max_pool_size,
            num_workers=num_workers
        )
        self.data_pool.start()
        self.buffer = []
        
    def __iter__(self):
        return self
        
    def __next__(self):
        # 如果缓冲区还有批次，直接返回
        if self.buffer:
            return self.buffer.pop(0)
            
        # 否则处理更多数据直到形成批次
        while True:
            if self.data_pool.data_queue.empty():
                time.sleep(0.1)  # 等待数据加载
                continue
                
            item = self.data_pool.data_queue.get()
            self.data_pool.data_queue.task_done()
            
            batch = self.batch_generator.batch_add_item(item)
            if batch is not None:
                return batch
                
    def __del__(self):
        self.data_pool.stop()