import queue
import threading
import time
import logging
import gc
import weakref
from typing import List, Dict, Any
import torch
from tqdm.auto import tqdm

class DataPrefetchPool:
    """
    内存优化的数据预加载池，使用生产者-消费者模式实现数据的异步加载，
    并包含显式内存管理机制
    """
    def __init__(self, dataset, max_size=100, num_workers=2, collate_fn=None, 
                 memory_cleanup_interval=200, enable_memory_tracking=True):
        """
        初始化数据预加载池
        
        Args:
            dataset: 数据集对象
            max_size: 数据池最大容量
            num_workers: 数据加载线程数
            collate_fn: 数据整理函数
            memory_cleanup_interval: 内存清理间隔（处理项目数）
            enable_memory_tracking: 是否启用内存使用跟踪
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
        
        # 内存管理相关
        self.memory_cleanup_interval = memory_cleanup_interval
        self.processed_items_count = 0
        self.enable_memory_tracking = enable_memory_tracking
        if enable_memory_tracking:
            self.initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            self.memory_stats = []
        
        # 弱引用集合，用于跟踪已经获取但可能尚未被Python垃圾收集的数据项
        self.item_refs = weakref.WeakSet()
        
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

    def get_item(self):
        # item = data_pool.data_queue.get()
        item = self.data_queue.get(block=True, timeout=5)
        self.item_refs.add(item)
        self.data_queue.task_done()
        self.processed_items_count += 1
        if self.processed_items_count >= self.memory_cleanup_interval:
            self._cleanup_memory()
            self.processed_items_count = 0
        
        return item
    
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
                # 跟踪处理的项目以便后续可能的清理
                self.item_refs.add(item)
                self.data_queue.task_done()
            except queue.Empty:
                if len(batch) > 0:
                    break
                else:
                    logging.warning("Timeout waiting for data")
        
        # 更新处理的项目计数并考虑清理内存
        self.processed_items_count += len(batch)
        if self.processed_items_count >= self.memory_cleanup_interval:
            self._cleanup_memory()
            self.processed_items_count = 0
        
        if self.collate_fn is not None and len(batch) > 0:
            collated_batch = self.collate_fn(batch)
            # 释放原始批次数据的引用
            batch = None
            return collated_batch
        return batch
    
    def _cleanup_memory(self):
        """显式清理内存"""
        # 记录清理前的内存使用情况
        # before_cleanup = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # 清除weakref集合中无效的引用
        self.item_refs.clear()
        
        # 显式触发Python垃圾收集
        gc.collect()
        
        # 如果使用CUDA，尝试释放缓存
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        
        # 记录清理后的内存使用情况
        # after_cleanup = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # if self.enable_memory_tracking:
            # freed = before_cleanup - after_cleanup
            # self.memory_stats.append((before_cleanup, after_cleanup, freed))
            # if freed > 0:
                # logging.info(f"Memory cleanup: freed {freed / (1024 * 1024):.2f} MB")
    
    def stop(self):
        """停止数据加载"""
        self.stop_event.set()
        for worker in self.workers:
            worker.join(timeout=2)
        
        # 最终清理
        self._cleanup_memory()
        logging.info("Data prefetch pool stopped")
            
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        
    def get_status(self):
        """获取数据池状态信息"""
        status = {
            "queue_size": self.data_queue.qsize(),
            "max_size": self.data_queue.maxsize,
            "workers": len(self.workers),
            "current_index": self.current_index,
            "dataset_size": self.dataset_size,
            "processed_since_cleanup": self.processed_items_count
        }
        
        # 添加内存使用情况
        if self.enable_memory_tracking and torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated()
            status["current_memory_mb"] = current_memory / (1024 * 1024)
            status["memory_change_mb"] = (current_memory - self.initial_memory) / (1024 * 1024)
            
        return status