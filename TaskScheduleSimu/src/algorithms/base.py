from abc import ABC, abstractmethod

class BaseScheduler(ABC):
    """
    调度算法基类
    """
    def __init__(self, problem):
        """
        初始化调度器
        
        参数:
        problem: 问题模型实例
        """
        self.problem = problem
    
    @abstractmethod
    def solve(self):
        """
        求解调度问题
        
        返回:
        调度结果
        """
        pass
