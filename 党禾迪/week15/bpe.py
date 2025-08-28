import re
import json
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set

class BPE:
    """Byte Pair Encoding (BPE) 实现类
    
    用于构建词表和进行文本的序列化/反序列化
    """
    
    def __init__(self):
        self.vocab = {}  # 词汇表：token -> id
        self.reverse_vocab = {}  # 反向词汇表：id -> token
        self.merges = []  # 合并规则列表
        self.word_freqs = {}  # 词频统计
        
    def get_word_tokens(self, text: str) -> Dict[str, int]:
        """将文本分割为单词并统计频率
        
        Args:
            text: 输入文本
            
        Returns:
            单词频率字典
        """
        # 使用正则表达式分割单词，保留标点符号
        words = re.findall(r'\w+|[^\w\s]', text.lower())
        word_freqs = Counter(words)
        return dict(word_freqs)
    
    def get_char_vocab(self, word_freqs: Dict[str, int]) -> Set[str]:
        """获取字符级别的初始词汇表
        
        Args:
            word_freqs: 单词频率字典
            
        Returns:
            字符集合
        """
        chars = set()
        for word in word_freqs.keys():
            chars.update(list(word))
        return chars
    
    def get_pairs(self, word_freqs: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """获取所有相邻字符对及其频率
        
        Args:
            word_freqs: 单词频率字典（单词已经被分割为字符序列）
            
        Returns:
            字符对频率字典
        """
        pairs = defaultdict(int)
        
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
                
        return dict(pairs)
    
    def merge_vocab(self, pair: Tuple[str, str], word_freqs: Dict[str, int]) -> Dict[str, int]:
        """合并指定的字符对
        
        Args:
            pair: 要合并的字符对
            word_freqs: 当前的单词频率字典
            
        Returns:
            合并后的单词频率字典
        """
        new_word_freqs = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        
        for word in word_freqs:
            new_word = p.sub(''.join(pair), word)
            new_word_freqs[new_word] = word_freqs[word]
            
        return new_word_freqs
    
    def build_vocab(self, text: str, vocab_size: int = 1000) -> None:
        """构建BPE词汇表
        
        Args:
            text: 训练文本
            vocab_size: 目标词汇表大小
        """
        print(f"开始构建BPE词汇表，目标大小: {vocab_size}")
        
        # 1. 获取单词频率
        word_freqs = self.get_word_tokens(text)
        print(f"发现 {len(word_freqs)} 个不同的单词")
        
        # 2. 将单词分割为字符序列
        word_freqs = {' '.join(list(word)): freq for word, freq in word_freqs.items()}
        
        # 3. 获取初始字符词汇表
        vocab = self.get_char_vocab(word_freqs)
        print(f"初始字符词汇表大小: {len(vocab)}")
        
        # 4. 迭代合并最频繁的字符对
        merges = []
        
        while len(vocab) < vocab_size:
            pairs = self.get_pairs(word_freqs)
            if not pairs:
                break
                
            # 找到频率最高的字符对
            best_pair = max(pairs, key=pairs.get)
            
            # 合并字符对
            word_freqs = self.merge_vocab(best_pair, word_freqs)
            merges.append(best_pair)
            vocab.add(''.join(best_pair))
            
            if len(vocab) % 100 == 0:
                print(f"当前词汇表大小: {len(vocab)}")
        
        # 5. 构建最终的词汇表映射
        self.vocab = {token: i for i, token in enumerate(sorted(vocab))}
        self.reverse_vocab = {i: token for token, i in self.vocab.items()}
        self.merges = merges
        self.word_freqs = word_freqs
        
        print(f"词汇表构建完成，最终大小: {len(self.vocab)}")
        print(f"合并规则数量: {len(self.merges)}")
    
    def apply_merges(self, word: str) -> List[str]:
        """对单个单词应用BPE合并规则
        
        Args:
            word: 输入单词
            
        Returns:
            BPE分割后的token列表
        """
        if not word:
            return []
            
        # 将单词分割为字符
        word_tokens = list(word)
        
        # 应用所有合并规则
        for pair in self.merges:
            i = 0
            while i < len(word_tokens) - 1:
                if word_tokens[i] == pair[0] and word_tokens[i + 1] == pair[1]:
                    # 合并这一对
                    word_tokens = word_tokens[:i] + [pair[0] + pair[1]] + word_tokens[i + 2:]
                else:
                    i += 1
        
        return word_tokens
    
    def encode(self, text: str) -> List[int]:
        """将文本编码为token ID序列
        
        Args:
            text: 输入文本
            
        Returns:
            token ID列表
        """
        if not self.vocab:
            raise ValueError("词汇表未构建，请先调用build_vocab方法")
        
        # 分割文本为单词
        words = re.findall(r'\w+|[^\w\s]', text.lower())
        
        token_ids = []
        for word in words:
            # 对每个单词应用BPE
            word_tokens = self.apply_merges(word)
            
            # 将token转换为ID
            for token in word_tokens:
                if token in self.vocab:
                    token_ids.append(self.vocab[token])
                else:
                    # 处理未知token，使用字符级别的回退
                    for char in token:
                        if char in self.vocab:
                            token_ids.append(self.vocab[char])
                        # 如果字符也不在词汇表中，跳过（或者可以使用特殊的UNK token）
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """将token ID序列解码为文本
        
        Args:
            token_ids: token ID列表
            
        Returns:
            解码后的文本
        """
        if not self.reverse_vocab:
            raise ValueError("词汇表未构建，请先调用build_vocab方法")
        
        tokens = []
        for token_id in token_ids:
            if token_id in self.reverse_vocab:
                tokens.append(self.reverse_vocab[token_id])
        
        # 简单地连接所有token（实际应用中可能需要更复杂的处理）
        return ''.join(tokens)
    
    def save_vocab(self, vocab_path: str, merges_path: str) -> None:
        """保存词汇表和合并规则到文件
        
        Args:
            vocab_path: 词汇表文件路径
            merges_path: 合并规则文件路径
        """
        # 保存词汇表
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        
        # 保存合并规则
        with open(merges_path, 'w', encoding='utf-8') as f:
            for pair in self.merges:
                f.write(f"{pair[0]} {pair[1]}\n")
        
        print(f"词汇表已保存到: {vocab_path}")
        print(f"合并规则已保存到: {merges_path}")
    
    def load_vocab(self, vocab_path: str, merges_path: str) -> None:
        """从文件加载词汇表和合并规则
        
        Args:
            vocab_path: 词汇表文件路径
            merges_path: 合并规则文件路径
        """
        # 加载词汇表
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # 加载合并规则
        self.merges = []
        with open(merges_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) == 2:
                        self.merges.append((parts[0], parts[1]))
        
        print(f"词汇表已加载，大小: {len(self.vocab)}")
        print(f"合并规则已加载，数量: {len(self.merges)}")
    
    def get_vocab_info(self) -> Dict:
        """获取词汇表信息
        
        Returns:
            词汇表信息字典
        """
        return {
            'vocab_size': len(self.vocab),
            'num_merges': len(self.merges),
            'sample_tokens': list(self.vocab.keys())[:10] if self.vocab else []
        }
