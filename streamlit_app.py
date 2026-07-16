import streamlit as st
import pandas as pd
import numpy as np
import re
import logging
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import io
import warnings
import time
warnings.filterwarnings('ignore')

# 设置页面
st.set_page_config(
    page_title="智能彩票分析检测系统",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 配置类 ====================
class BaccaratConfig:
    def __init__(self):
        self.min_amount = 10  # 最小投注金额阈值
        self.amount_similarity_threshold = 0.8  # 金额相似度阈值
        self.min_continuous_periods = 3  # 最小连续局数
        self.max_accounts_in_group = 5  # 最大检测账户数
        
        # 列名映射配置 - 百家乐专用
        self.column_mappings = {
            '会员账号': [
                '会员账号', '会员账户', '账号', '账户', '用户账号', '玩家账号', '用户ID', '玩家ID', 
                '会员名', '用户名', '玩家名称', '用户名称', 'Account', 'account', '会员',
                '玩家', '用户', '会员帐号', '会员编码', '会员号', '用户号', '玩家号',
                '会员ID', '玩家ID', '用户ID', '帐号', '帐戶', '账户名', '账号名'
            ],
            '局号': [
                '局号', '局数', '局次', '游戏局号', '牌局号', '期号', '奖期', '期数', '游戏期号',
                'Round', 'round', '局', '遊戲局號', '牌局編號', '游戏编号', '局编号',
                '牌局', '遊戲局', '游戏局', '局次号', '桌号', '桌台号', '台号'
            ],
            '游戏类型': [
                '游戏类型', '游戏种类', '游戏名称', '游戏', '彩种', '彩票类型', '游戏分类',
                'Game Type', 'game', 'game_type', 'Game', 'GAME', '遊戲類型', '遊戲種類',
                '游戏模式', '游戏分类', '游戏系列', '产品类型', '产品', '遊戲', '產品類型'
            ],
            '下注玩法': [
                '下注玩法', '玩法', '投注类型', '投注玩法', '玩法分类', '类型', '投注方向', '下注类型',
                'Bet Type', 'bet_type', 'bet', 'Bet', '玩法類型', '投注種類', '投注項目',
                '投注内容', '下注内容', '下注項目', '注单类型', '注单玩法', '注码类型'
            ],
            '下注额度': [
                '下注额度', '金额', '投注金额', '下注金额', '投注额', '额度', '投注额度', '下注额',
                'Amount', 'amount', 'bet_amount', 'money', 'Money', '下注金額', '投注金額',
                '投注量', '下注量', '注额', '注碼', '筹码', '筹码额', '投注筹码', '下注筹码'
            ]
        }
        
        # 百家乐游戏类型变体
        self.game_type_variants = {
            '百家乐': [
                '百家乐', 'Baccarat', 'bac', 'BJL', 'bjl', '真人百家乐', '视讯百家乐', 
                'BBIN百家乐', 'AG百家乐', '欧博百家乐', '申博百家乐', 'WM百家乐',
                '百家樂', 'Baccarat', 'バカラ', '바카라', '바카라', '바카라게임',
                '经典百家乐', '急速百家乐', '闪电百家乐', '多台百家乐', 'VIP百家乐'
            ],
            '龙虎': [
                '龙虎', '龙虎斗', 'Dragon Tiger', '龙虎大战', '龙虎游戏', '龍虎', '龍虎鬥',
                '龙虎斗游戏', 'Dragon Tiger Game', 'ドラゴンタイガー', '드래곤타이거'
            ]
        }
        
        # 百家乐下注玩法变体映射
        self.bet_type_variants = {
            # 庄闲类
            '庄': [
                '庄', '莊', 'Banker', 'banker', '庄家', '莊家', 'B', 'b', '庄赢', '莊贏', '庄胜', '莊勝',
                '庄方', '庄闲-庄', '庄闲_庄', '庄-庄', 'bank', 'Bank', '庄庄', '莊莊', '庄家赢', '莊家贏',
                '庄家胜', 'Banker Win', 'banker win', '庄赢家', 'Banker Side', '庄侧', '庄边'
            ],
            '闲': [
                '闲', '閒', 'Player', 'player', '闲家', '閒家', 'P', 'p', '闲赢', '閒贏', '闲胜', '閒勝',
                '闲方', '庄闲-闲', '庄闲_闲', '庄-闲', 'play', 'Play', '闲闲', '閒閒', '闲家赢', '閒家贏',
                '闲家胜', 'Player Win', 'player win', '闲赢家', 'Player Side', '闲侧', '闲边'
            ],
            '和': [
                '和', '和局', 'Tie', 'tie', '平局', '平', 'T', 't', '和棋', '和牌', '和赢', '和勝',
                '庄闲-和', '庄闲_和', '庄-和', 'TIE', 'Tie Win', 'tie win', '平局赢', '和局赢',
                '和赢家', '平赢', '平胜'
            ],
            '庄对': [
                '庄对', '莊對', 'Banker Pair', 'banker pair', '庄对子', '莊對子', 'BP', 'bp',
                '庄对子', '庄对子胜', '庄对子赢', '庄对子_庄', '庄对-庄', 'Banker Pair Win',
                '庄对赢', '庄对胜', '庄家对子', '莊家對子'
            ],
            '闲对': [
                '闲对', '閒對', 'Player Pair', 'player pair', '闲对子', '閒對子', 'PP', 'pp',
                '闲对子', '闲对子胜', '闲对子赢', '闲对子_闲', '闲对-闲', 'Player Pair Win',
                '闲对赢', '闲对胜', '闲家对子', '閒家對子'
            ],
            
            # 龙虎类
            '龙': [
                '龙', '龍', 'Dragon', 'dragon', '龙方', '龍方', 'D', 'd', '龙赢', '龍贏', '龙胜', '龍勝',
                '龙虎-龙', '龙虎_龙', '龙虎龙', '龙虎鬥-龙', '龙-龙', 'Dragon Win', 'dragon win',
                '龙赢家', '龙方赢', '龙边', '龙侧'
            ],
            '虎': [
                '虎', 'Tiger', 'tiger', '虎方', 'T', 't', '虎赢', '虎勝', '虎方赢', '虎方勝',
                '龙虎-虎', '龙虎_虎', '龙虎虎', '龙虎鬥-虎', '虎-虎', 'Tiger Win', 'tiger win',
                '虎赢家', '虎方赢', '虎边', '虎侧'
            ],
            
            # 其他玩法（可选检测）
            '大': ['大', 'Big', 'big', '大牌', '大点', '大數', '大-大', '大小-大', '大小_大'],
            '小': ['小', 'Small', 'small', '小牌', '小点', '小數', '小-小', '大小-小', '大小_小'],
            '单': ['单', '單', 'Odd', 'odd', '单数', '單數', '奇数', '奇數', '单-单', '单双-单', '单双_单'],
            '双': ['双', '雙', 'Even', 'even', '双数', '雙數', '偶数', '偶數', '双-双', '单双-双', '单双_双']
        }
        
        # 对立组配置 - 百家乐核心对立关系
        self.opposite_groups = [
            {'庄', '闲'},  # 庄 vs 闲 (核心对刷)
            {'龙', '虎'},  # 龙 vs 虎 (核心对刷)
            {'庄', '和'},  # 庄 vs 和 (次要对刷)
            {'闲', '和'},  # 闲 vs 和 (次要对刷)
            {'庄对', '闲对'},  # 庄对 vs 闲对
            {'大', '小'},  # 大 vs 小
            {'单', '双'}   # 单 vs 双
        ]
        
        # 单账户禁止同时下注的组合
        self.exclusive_bet_groups = [
            {'庄', '闲'},  # 不能同时下注庄和闲
            {'龙', '虎'},  # 不能同时下注龙和虎
            {'大', '小'},  # 不能同时下注大和小
            {'单', '双'},  # 不能同时下注单和双
            {'庄对', '闲对'}  # 通常不同时下注庄对和闲对
        ]
        
        # 活跃度阈值配置
        self.period_thresholds = {
            'low_activity': 10,          # 低活跃度阈值上限
            'medium_activity_low': 11,   # 中活跃度阈值下限
            'medium_activity_high': 50,  # 中活跃度阈值上限
            'high_activity_low': 51,     # 高活跃度阈值下限
            'high_activity_high': 100,   # 高活跃度阈值上限
            'min_periods_low': 3,        # 低活跃度最小连续局数
            'min_periods_medium': 5,     # 中活跃度最小连续局数
            'min_periods_high': 8,       # 高活跃度最小连续局数
            'min_periods_very_high': 11  # 极高活跃度最小连续局数
        }
        
        # 多账户匹配度阈值
        self.account_count_similarity_thresholds = {
            2: 0.8,   # 2个账户
            3: 0.85,  # 3个账户
            4: 0.9,   # 4个账户
            5: 0.95   # 5个账户
        }
        
        # 账户局数差异阈值
        self.account_period_diff_threshold = 101
        
        # 金额阈值配置
        self.amount_threshold = {
            'max_amount_ratio': 10,  # 最大金额比例
            'enable_threshold_filter': True  # 启用金额阈值过滤
        }

# ==================== 数据处理器类 ====================
class BaccaratDataProcessor:
    def __init__(self):
        self.required_columns = ['会员账号', '局号', '游戏类型', '下注玩法', '下注额度']
        self.config = BaccaratConfig()
        self.similarity_threshold = 0.7
        
        # 百家乐特定关键词
        self.baccarat_keywords = ['百家乐', 'Baccarat', 'bac', 'BJL', 'bjl', '百家樂']
        self.dragon_tiger_keywords = ['龙虎', '龙虎斗', 'Dragon Tiger', '龍虎', '龍虎鬥']
    
    def smart_column_identification(self, df_columns):
        """智能列识别 - 支持百家乐数据"""
        identified_columns = {}
        actual_columns = [str(col).strip() for col in df_columns]
        
        for standard_col, possible_names in self.config.column_mappings.items():
            found = False
            for actual_col in actual_columns:
                actual_col_lower = actual_col.lower().replace(' ', '').replace('_', '').replace('-', '')
                
                for possible_name in possible_names:
                    possible_name_lower = possible_name.lower().replace(' ', '').replace('_', '').replace('-', '')
                    
                    # 计算相似度
                    set1 = set(possible_name_lower)
                    set2 = set(actual_col_lower)
                    intersection = set1 & set2
                    
                    similarity_score = len(intersection) / len(set1) if set1 else 0
                    
                    # 匹配条件
                    if (possible_name_lower in actual_col_lower or 
                        actual_col_lower in possible_name_lower or
                        similarity_score >= self.similarity_threshold):
                        
                        identified_columns[actual_col] = standard_col
                        found = True
                        break
                
                if found:
                    break
            
            # 如果没有找到匹配，但还有标准列未匹配，尝试模糊匹配
            if not found and standard_col not in identified_columns.values():
                for actual_col in actual_columns:
                    actual_col_lower = actual_col.lower()
                    if standard_col in actual_col_lower or any(keyword in actual_col_lower for keyword in standard_col.lower()):
                        identified_columns[actual_col] = standard_col
                        break
        
        return identified_columns
    
    def find_data_start(self, df):
        """智能找到数据起始位置 - 针对百家乐数据"""
        for row_idx in range(min(30, len(df))):
            for col_idx in range(min(10, len(df.columns))):
                cell_value = str(df.iloc[row_idx, col_idx])
                if pd.notna(cell_value):
                    # 检查是否是表头关键词
                    if any(keyword in cell_value for keyword in ['会员', '账号', '账户', '用户', '玩家', 'Account']):
                        return row_idx, col_idx
                    # 检查是否是数据行（包含局号或金额）
                    if any(keyword in cell_value for keyword in ['局', 'Round', 'round', '金额', 'amount', 'Amount']):
                        return max(0, row_idx-1), col_idx
        
        return 0, 0
    
    def validate_data_quality(self, df):
        """数据质量验证"""
        logger.info("正在进行数据质量验证...")
        issues = []
        
        # 检查必要列
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            issues.append(f"缺少必要列: {missing_cols}")
        
        # 检查空值
        for col in self.required_columns:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    issues.append(f"列 '{col}' 有 {null_count} 个空值")
        
        # 检查重复数据
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            issues.append(f"发现 {duplicate_count} 条重复记录")
        
        # 检查游戏类型有效性
        if '游戏类型' in df.columns:
            invalid_game_types = df[~df['游戏类型'].astype(str).str.contains('|'.join(['百家乐', '龙虎', 'Baccarat', 'Dragon']), na=False)]
            if len(invalid_game_types) > 0:
                issues.append(f"发现 {len(invalid_game_types)} 条非百家乐/龙虎游戏记录")
        
        return issues
    
    def clean_data(self, uploaded_file):
        """数据清洗主函数 - 专为百家乐设计"""
        try:
            # 尝试读取文件判断编码和格式
            if uploaded_file.name.endswith('.csv'):
                # 尝试不同编码读取CSV
                encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
                df_temp = None
                for encoding in encodings:
                    try:
                        df_temp = pd.read_csv(uploaded_file, encoding=encoding, nrows=50, header=None)
                        break
                    except:
                        continue
                if df_temp is None:
                    st.error("❌ 无法读取CSV文件，请检查编码格式")
                    return None
            else:
                # Excel文件
                df_temp = pd.read_excel(uploaded_file, header=None, nrows=50)
            
            # 找到数据起始位置
            start_row, start_col = self.find_data_start(df_temp)
            
            # 重新读取完整数据
            if uploaded_file.name.endswith('.csv'):
                df_clean = pd.read_csv(
                    uploaded_file, 
                    encoding='utf-8',
                    header=start_row,
                    skiprows=range(start_row + 1) if start_row > 0 else None,
                    dtype=str,
                    na_filter=False,
                    engine='python'
                )
            else:
                df_clean = pd.read_excel(
                    uploaded_file, 
                    header=start_row,
                    skiprows=range(start_row + 1) if start_row > 0 else None,
                    dtype=str,
                    na_filter=False
                )
            
            # 裁剪列
            if start_col > 0:
                df_clean = df_clean.iloc[:, start_col:]
            
            # 智能列识别
            column_mapping = self.smart_column_identification(df_clean.columns)
            if column_mapping:
                df_clean = df_clean.rename(columns=column_mapping)
            
            # 如果仍有缺失列，尝试自动分配（更智能的映射）
            missing_columns = [col for col in self.required_columns if col not in df_clean.columns]
            if missing_columns:
                # 尝试根据列名特征匹配
                for col in df_clean.columns:
                    col_lower = str(col).lower()
                    if '会员' in col_lower or '账号' in col_lower or 'account' in col_lower:
                        df_clean = df_clean.rename(columns={col: '会员账号'})
                    elif '局' in col_lower or 'round' in col_lower:
                        df_clean = df_clean.rename(columns={col: '局号'})
                    elif '游戏' in col_lower or 'game' in col_lower:
                        df_clean = df_clean.rename(columns={col: '游戏类型'})
                    elif '玩法' in col_lower or 'bet' in col_lower or '下注' in col_lower:
                        df_clean = df_clean.rename(columns={col: '下注玩法'})
                    elif '金额' in col_lower or 'amount' in col_lower or '额度' in col_lower:
                        df_clean = df_clean.rename(columns={col: '下注额度'})
            
            # 再次检查必要列
            missing_columns = [col for col in self.required_columns if col not in df_clean.columns]
            if missing_columns and len(df_clean.columns) >= 5:
                # 按位置分配
                col_positions = {0: '会员账号', 1: '局号', 2: '游戏类型', 3: '下注玩法', 4: '下注额度'}
                for i, col in enumerate(df_clean.columns[:5]):
                    if i in col_positions:
                        df_clean = df_clean.rename(columns={col: col_positions[i]})
            
            # 删除完全空白的行和列
            initial_count = len(df_clean)
            df_clean = df_clean.dropna(how='all')
            df_clean = df_clean.dropna(axis=1, how='all')
            
            # 确保必要列存在
            for col in self.required_columns:
                if col not in df_clean.columns:
                    df_clean[col] = ''
            
            # 数据格式化
            for col in self.required_columns:
                df_clean[col] = df_clean[col].astype(str).str.strip()
            
            # 特殊处理局号列
            if '局号' in df_clean.columns:
                df_clean['局号'] = df_clean['局号'].str.replace(r'\.0$', '', regex=True)
                df_clean['局号'] = df_clean['局号'].str.replace(r'\s+', '', regex=True)
            
            # 处理金额列
            if '下注额度' in df_clean.columns:
                df_clean['下注额度_原始'] = df_clean['下注额度']
                df_clean['下注额度'] = df_clean['下注额度'].apply(self.preprocess_amount_column)
            
            # 验证数据质量
            quality_issues = self.validate_data_quality(df_clean)
            if quality_issues:
                logger.warning(f"数据质量问题: {quality_issues}")
            
            # 过滤无效记录
            valid_mask = (
                (df_clean['会员账号'].str.len() > 0) &
                (df_clean['局号'].str.len() > 0) &
                (df_clean['游戏类型'].str.len() > 0) &
                (df_clean['下注玩法'].str.len() > 0)
            )
            df_clean = df_clean[valid_mask].copy()
            
            logger.info(f"数据清洗完成: {initial_count} -> {len(df_clean)} 条记录")
            
            return df_clean
                
        except Exception as e:
            st.error(f"❌ 数据清洗失败: {str(e)}")
            logger.error(f"数据清洗失败: {str(e)}")
            traceback.print_exc()
            return None
    
    def preprocess_amount_column(self, amount_text):
        """预处理金额列格式 - 专为百家乐设计"""
        if pd.isna(amount_text):
            return "0"
        
        text = str(amount_text).strip()
        
        # 处理各种金额格式
        # 1. 移除货币符号
        text = re.sub(r'[￥¥＄\$€£￡]', '', text)
        
        # 2. 处理中文冒号和英文冒号
        if '：' in text:
            parts = text.split('：')
            if len(parts) >= 2:
                text = parts[-1].strip()
        elif ':' in text:
            parts = text.split(':')
            if len(parts) >= 2:
                text = parts[-1].strip()
        
        # 3. 提取数字部分（支持小数和负数）
        numbers = re.findall(r'[-+]?\d*\.?\d+', text.replace(',', ''))
        if numbers:
            # 取最后一个数字（通常是最准确的）
            amount = numbers[-1]
            # 验证是否为有效金额
            try:
                float_amount = float(amount)
                if float_amount > 0:
                    return amount
            except:
                pass
        
        return "0"

# ==================== 游戏类型识别器 ====================
class GameTypeIdentifier:
    def __init__(self):
        self.config = BaccaratConfig()
        self.game_type_mapping = {}
        
        # 构建游戏类型映射
        for standard_name, variants in self.config.game_type_variants.items():
            for variant in variants:
                self.game_type_mapping[variant.lower()] = standard_name
    
    def identify_game_type(self, game_type_text):
        """识别游戏类型 - 专为百家乐设计"""
        if pd.isna(game_type_text):
            return "未知"
        
        text = str(game_type_text).strip()
        text_lower = text.lower()
        
        # 精确匹配
        if text_lower in self.game_type_mapping:
            return self.game_type_mapping[text_lower]
        
        # 模糊匹配
        for standard_name, variants in self.config.game_type_variants.items():
            for variant in variants:
                variant_lower = variant.lower()
                if variant_lower in text_lower or text_lower in variant_lower:
                    return standard_name
        
        # 根据关键词判断
        if any(keyword in text_lower for keyword in ['baccarat', '百家乐', 'bjl']):
            return '百家乐'
        elif any(keyword in text_lower for keyword in ['dragon', 'tiger', '龙虎']):
            return '龙虎'
        
        return text

# ==================== 下注玩法标准化器 ====================
class BetTypeNormalizer:
    def __init__(self):
        self.config = BaccaratConfig()
        self.bet_type_mapping = {}
        
        # 构建下注玩法映射
        for standard_name, variants in self.config.bet_type_variants.items():
            for variant in variants:
                self.bet_type_mapping[variant.lower()] = standard_name
    
    def normalize_bet_type(self, bet_type_text):
        """标准化下注玩法 - 专为百家乐设计"""
        if pd.isna(bet_type_text):
            return ""
        
        text = str(bet_type_text).strip()
        text_lower = text.lower()
        
        # 精确匹配
        if text_lower in self.bet_type_mapping:
            return self.bet_type_mapping[text_lower]
        
        # 模糊匹配
        for standard_name, variants in self.config.bet_type_variants.items():
            for variant in variants:
                variant_lower = variant.lower()
                if variant_lower in text_lower or text_lower in variant_lower:
                    return standard_name
        
        # 根据关键词判断
        keyword_mapping = [
            (['庄', 'banker', 'bank'], '庄'),
            (['闲', 'player', 'play'], '闲'),
            (['和', 'tie', '平'], '和'),
            (['龙', 'dragon'], '龙'),
            (['虎', 'tiger'], '虎'),
            (['庄对', 'banker pair'], '庄对'),
            (['闲对', 'player pair'], '闲对'),
            (['大', 'big'], '大'),
            (['小', 'small'], '小'),
            (['单', 'odd'], '单'),
            (['双', 'even'], '双')
        ]
        
        for keywords, bet_type in keyword_mapping:
            if any(keyword in text_lower for keyword in keywords):
                return bet_type
        
        return text

# ==================== 百家乐对刷检测器类 ====================
class BaccaratWashTradeDetector:
    def __init__(self, config=None):
        self.config = config or BaccaratConfig()
        self.data_processor = BaccaratDataProcessor()
        self.game_type_identifier = GameTypeIdentifier()
        self.bet_type_normalizer = BetTypeNormalizer()
        
        self.data_processed = False
        self.df_valid = None
        self.export_data = []
        
        # 统计信息
        self.account_total_periods_by_game = defaultdict(dict)
        self.account_record_stats_by_game = defaultdict(dict)
        self.performance_stats = {}
    
    def upload_and_process(self, uploaded_file):
        """上传并处理文件"""
        try:
            if uploaded_file is None:
                st.error("❌ 没有上传文件")
                return None, None
            
            filename = uploaded_file.name
            logger.info(f"✅ 已上传文件: {filename}")
            
            # 检查文件类型
            supported_types = ['.xlsx', '.xls', '.csv']
            if not any(filename.endswith(ext) for ext in supported_types):
                st.error(f"❌ 不支持的文件类型: {filename}")
                return None, None
            
            # 清洗数据
            with st.spinner("🔄 正在清洗数据..."):
                df_clean = self.data_processor.clean_data(uploaded_file)
            
            if df_clean is not None and len(df_clean) > 0:
                # 增强数据处理
                df_enhanced = self.enhance_data_processing(df_clean)
                return df_enhanced, filename
            else:
                return None, None
            
        except Exception as e:
            logger.error(f"文件处理失败: {str(e)}")
            st.error(f"文件处理失败: {str(e)}")
            traceback.print_exc()
            return None, None
    
    def enhance_data_processing(self, df_clean):
        """增强数据处理流程 - 专为百家乐设计"""
        try:
            # 1. 标准化游戏类型
            if '游戏类型' in df_clean.columns:
                df_clean['标准化游戏类型'] = df_clean['游戏类型'].apply(
                    self.game_type_identifier.identify_game_type
                )
            else:
                df_clean['标准化游戏类型'] = '未知'
            
            # 2. 标准化下注玩法
            if '下注玩法' in df_clean.columns:
                df_clean['标准化下注玩法'] = df_clean['下注玩法'].apply(
                    self.bet_type_normalizer.normalize_bet_type
                )
            else:
                df_clean['标准化下注玩法'] = ''
            
            # 3. 提取投注金额（数值型）
            if '下注额度' in df_clean.columns:
                df_clean['投注金额'] = df_clean['下注额度'].apply(
                    lambda x: float(x) if self._is_float(x) else 0.0
                )
            else:
                df_clean['投注金额'] = 0.0
            
            # 4. 过滤有效数据
            df_valid = df_clean[
                (df_clean['标准化下注玩法'].isin(['庄', '闲', '和', '龙', '虎', '庄对', '闲对'])) & 
                (df_clean['投注金额'] >= self.config.min_amount) &
                (df_clean['标准化游戏类型'].isin(['百家乐', '龙虎']))
            ].copy()
            
            # 5. 添加辅助列
            df_valid['检测标识'] = df_valid.apply(
                lambda row: f"{row['会员账号']}_{row['局号']}_{row['标准化游戏类型']}", 
                axis=1
            )
            
            self.data_processed = True
            self.df_valid = df_valid
            
            # 6. 计算账户统计信息
            self.calculate_account_total_periods_by_game(df_valid)
            
            logger.info(f"数据处理完成: {len(df_clean)} -> {len(df_valid)} 条有效记录")
            
            # 显示数据预览
            with st.expander("📊 数据预览", expanded=False):
                st.write(f"**数据统计:**")
                st.write(f"- 总记录数: {len(df_clean):,}")
                st.write(f"- 有效记录数: {len(df_valid):,}")
                st.write(f"- 唯一账户数: {df_valid['会员账号'].nunique():,}")
                st.write(f"- 唯一局号数: {df_valid['局号'].nunique():,}")
                
                if len(df_valid) > 0:
                    st.write(f"**游戏类型分布:**")
                    game_stats = df_valid['标准化游戏类型'].value_counts()
                    for game, count in game_stats.items():
                        st.write(f"  - {game}: {count:,} 条记录")
                    
                    st.write(f"**下注玩法分布:**")
                    bet_stats = df_valid['标准化下注玩法'].value_counts()
                    for bet, count in bet_stats.items():
                        st.write(f"  - {bet}: {count:,} 条记录")
            
            return df_valid
                
        except Exception as e:
            logger.error(f"数据处理增强失败: {str(e)}")
            st.error(f"数据处理增强失败: {str(e)}")
            traceback.print_exc()
            return pd.DataFrame()
    
    def _is_float(self, value):
        """检查是否为浮点数"""
        try:
            float(value)
            return True
        except:
            return False
    
    def calculate_account_total_periods_by_game(self, df):
        """计算账户在每种游戏上的总局数"""
        self.account_total_periods_by_game = defaultdict(dict)
        self.account_record_stats_by_game = defaultdict(dict)
        
        # 确保使用有效数据
        data_source = self.df_valid if hasattr(self, 'df_valid') and self.df_valid is not None else df
        
        if '标准化游戏类型' not in data_source.columns:
            return
        
        # 按游戏类型分组统计
        for game_type in data_source['标准化游戏类型'].unique():
            df_game = data_source[data_source['标准化游戏类型'] == game_type]
            
            # 统计每个账户的局数
            period_counts = df_game.groupby('会员账号')['局号'].nunique().to_dict()
            self.account_total_periods_by_game[game_type] = period_counts
            
            # 统计每个账户的记录数
            record_counts = df_game.groupby('会员账号').size().to_dict()
            self.account_record_stats_by_game[game_type] = record_counts
    
    def detect_all_wash_trades(self):
        """主检测方法：检测所有对刷模式"""
        if not self.data_processed or self.df_valid is None or len(self.df_valid) == 0:
            st.error("❌ 没有有效数据可用于检测")
            return []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_patterns = []
        total_steps = self.config.max_accounts_in_group + 2
        
        # 1. 检测单账户对刷（同一账户在同一局下注对立面）
        status_text.text("🔍 检测单账户对刷模式...")
        single_account_patterns = self.detect_single_account_wash_trades(self.df_valid)
        all_patterns.extend(single_account_patterns)
        
        progress_bar.progress(1 / total_steps)
        
        # 2. 检测多账户对刷（多个账户在同一局下注对立面）
        for account_count in range(2, self.config.max_accounts_in_group + 1):
            status_text.text(f"🔍 检测{account_count}个账户对刷模式...")
            patterns = self.detect_multi_account_wash_trades(self.df_valid, account_count)
            all_patterns.extend(patterns)
            
            progress = (account_count) / total_steps
            progress_bar.progress(progress)
        
        progress_bar.progress(1.0)
        status_text.empty()
        
        # 3. 查找连续模式
        continuous_patterns = self.find_continuous_patterns(all_patterns)
        
        logger.info(f"检测完成: 发现 {len(continuous_patterns)} 个连续对刷模式")
        
        return continuous_patterns
    
    def detect_single_account_wash_trades(self, df):
        """检测单账户对刷模式（同一账户在同一局下注对立面）"""
        patterns = []
        
        # 按账户、局号、游戏类型分组
        grouped = df.groupby(['会员账号', '局号', '标准化游戏类型'])
        
        for (account, period, game_type), group in grouped:
            if len(group) < 2:
                continue
            
            # 获取该局的所有下注玩法
            bet_types = set(group['标准化下注玩法'].tolist())
            
            # 检查是否有对立的下注
            for opposite_group in self.config.exclusive_bet_groups:
                # 检查是否同时包含对立组中的两个元素
                if opposite_group.issubset(bet_types):
                    # 提取对立下注的详细信息
                    opposite_bets = []
                    total_amount = 0
                    
                    for bet_type in opposite_group:
                        bet_data = group[group['标准化下注玩法'] == bet_type]
                        if not bet_data.empty:
                            amount = bet_data['投注金额'].sum()
                            opposite_bets.append({
                                '下注玩法': bet_type,
                                '金额': amount
                            })
                            total_amount += amount
                    
                    if len(opposite_bets) == 2:
                        # 计算相似度（金额比例）
                        amount1 = opposite_bets[0]['金额']
                        amount2 = opposite_bets[1]['金额']
                        similarity = min(amount1, amount2) / max(amount1, amount2) if max(amount1, amount2) > 0 else 0
                        
                        if similarity >= self.config.amount_similarity_threshold:
                            pattern = {
                                '局号': period,
                                '游戏类型': game_type,
                                '账户组': [account],
                                '账户数量': 1,
                                '下注玩法组': [opposite_bets[0]['下注玩法'], opposite_bets[1]['下注玩法']],
                                '金额组': [amount1, amount2],
                                '总金额': total_amount,
                                '相似度': similarity,
                                '模式': f'单账户对立下注-{opposite_bets[0]["下注玩法"]}vs{opposite_bets[1]["下注玩法"]}',
                                '对立类型': f'{opposite_bets[0]["下注玩法"]}-{opposite_bets[1]["下注玩法"]}',
                                '检测类型': '单账户对刷',
                                '账户活跃度': self.get_account_activity_level(account, game_type)
                            }
                            patterns.append(pattern)
        
        logger.info(f"单账户对刷检测完成: 发现 {len(patterns)} 个模式")
        return patterns
    
    def detect_multi_account_wash_trades(self, df, n_accounts):
        """检测多账户对刷模式"""
        patterns = []
        
        # 按局号和游戏类型分组
        grouped = df.groupby(['局号', '标准化游戏类型'])
        
        for (period, game_type), period_data in grouped:
            accounts = period_data['会员账号'].unique()
            
            if len(accounts) < n_accounts:
                continue
            
            # 检查账户局数差异
            valid_accounts = []
            for account in accounts:
                if self._check_account_period_difference([account], game_type):
                    valid_accounts.append(account)
            
            if len(valid_accounts) < n_accounts:
                continue
            
            # 生成所有可能的账户组合
            for account_group in combinations(valid_accounts, n_accounts):
                group_patterns = self._analyze_account_group(period_data, list(account_group), period, game_type, n_accounts)
                patterns.extend(group_patterns)
        
        logger.info(f"多账户对刷检测完成({n_accounts}账户): 发现 {len(patterns)} 个模式")
        return patterns
    
    def _analyze_account_group(self, period_data, account_group, period, game_type, n_accounts):
        """分析账户组的对刷模式"""
        patterns = []
        
        # 获取账户组的数据
        group_data = period_data[period_data['会员账号'].isin(account_group)]
        
        if len(group_data) < n_accounts:
            return patterns
        
        # 分析每个账户的下注
        account_bets = {}
        for account in account_group:
            account_data = group_data[group_data['会员账号'] == account]
            if not account_data.empty:
                # 取该账户在该局的主要下注（金额最大的）
                main_bet = account_data.loc[account_data['投注金额'].idxmax()]
                account_bets[account] = {
                    '下注玩法': main_bet['标准化下注玩法'],
                    '金额': main_bet['投注金额']
                }
        
        if len(account_bets) < n_accounts:
            return patterns
        
        # 检查对立模式
        for opposite_group in self.config.opposite_groups:
            opposite_list = list(opposite_group)
            
            if n_accounts == 2:
                # 2个账户：一个下注A，一个下注B
                if len(opposite_list) == 2:
                    dir1, dir2 = opposite_list
                    
                    # 检查账户组的下注是否符合这种对立
                    bet_types = [account_bets[acc]['下注玩法'] for acc in account_group]
                    
                    if set(bet_types) == {dir1, dir2}:
                        # 金额相似度检查
                        amounts = [account_bets[acc]['金额'] for acc in account_group]
                        max_ratio = self.config.amount_threshold['max_amount_ratio']
                        
                        if max(amounts) / min(amounts) <= max_ratio:
                            similarity = min(amounts) / max(amounts) if max(amounts) > 0 else 0
                            
                            if similarity >= self.config.account_count_similarity_thresholds[n_accounts]:
                                pattern = {
                                    '局号': period,
                                    '游戏类型': game_type,
                                    '账户组': account_group,
                                    '账户数量': n_accounts,
                                    '下注玩法组': bet_types,
                                    '金额组': amounts,
                                    '总金额': sum(amounts),
                                    '相似度': similarity,
                                    '模式': f'多账户对立下注-{dir1}vs{dir2}',
                                    '对立类型': f'{dir1}-{dir2}',
                                    '检测类型': '多账户对刷'
                                }
                                patterns.append(pattern)
            
            elif n_accounts == 3:
                # 3个账户：例如2个下注庄，1个下注闲
                for dir1_count in range(1, n_accounts):
                    dir2_count = n_accounts - dir1_count
                    
                    # 检查账户组的下注分布
                    bet_types = [account_bets[acc]['下注玩法'] for acc in account_group]
                    
                    # 检查是否都是对立组中的玩法
                    if all(bt in opposite_list for bt in bet_types):
                        # 检查分布是否符合
                        dir1 = opposite_list[0]
                        dir2 = opposite_list[1] if len(opposite_list) > 1 else opposite_list[0]
                        
                        actual_dir1_count = bet_types.count(dir1)
                        actual_dir2_count = bet_types.count(dir2)
                        
                        if actual_dir1_count == dir1_count and actual_dir2_count == dir2_count:
                            # 金额检查
                            amounts = [account_bets[acc]['金额'] for acc in account_group]
                            
                            # 过滤金额差异过大的组合
                            filtered_accounts, filtered_amounts = self._filter_by_amount_balance(
                                account_group, amounts
                            )
                            
                            if len(filtered_accounts) >= 2:
                                similarity = self._calculate_similarity(filtered_amounts)
                                
                                if similarity >= self.config.account_count_similarity_thresholds[n_accounts]:
                                    pattern = {
                                        '局号': period,
                                        '游戏类型': game_type,
                                        '账户组': filtered_accounts,
                                        '账户数量': len(filtered_accounts),
                                        '下注玩法组': [account_bets[acc]['下注玩法'] for acc in filtered_accounts],
                                        '金额组': filtered_amounts,
                                        '总金额': sum(filtered_amounts),
                                        '相似度': similarity,
                                        '模式': f'多账户对立下注-{dir1}({dir1_count})vs{dir2}({dir2_count})',
                                        '对立类型': f'{dir1}-{dir2}',
                                        '检测类型': '多账户对刷'
                                    }
                                    patterns.append(pattern)
        
        return patterns
    
    def _check_account_period_difference(self, account_group, game_type):
        """检查账户组内账户的总投注局数差异是否在阈值内"""
        if game_type not in self.account_total_periods_by_game:
            return True
        
        total_periods_stats = self.account_total_periods_by_game[game_type]
        
        account_periods = []
        for account in account_group:
            if account in total_periods_stats:
                account_periods.append(total_periods_stats[account])
            else:
                return True
        
        if len(account_periods) < 2:
            return True
        
        max_period = max(account_periods)
        min_period = min(account_periods)
        period_diff = max_period - min_period
        
        if period_diff > self.config.account_period_diff_threshold:
            return False
        
        return True
    
    def _filter_by_amount_balance(self, account_group, amounts):
        """根据金额平衡性过滤账户"""
        if not self.config.amount_threshold['enable_threshold_filter']:
            return account_group, amounts
        
        if not amounts or len(amounts) < 2:
            return account_group, amounts
        
        max_amount = max(amounts)
        min_amount = min(amounts)
        
        amount_ratio = max_amount / min_amount if min_amount > 0 else float('inf')
        max_allowed_ratio = self.config.amount_threshold['max_amount_ratio']
        
        if amount_ratio <= max_allowed_ratio:
            return account_group, amounts
        
        # 过滤金额差异过大的账户
        min_required = max_amount / max_allowed_ratio
        valid_indices = [i for i, amount in enumerate(amounts) if amount >= min_required]
        
        if len(valid_indices) >= 2:
            filtered_accounts = [account_group[i] for i in valid_indices]
            filtered_amounts = [amounts[i] for i in valid_indices]
            
            logger.debug(f"金额平衡过滤: {len(account_group)} -> {len(filtered_accounts)} 个账户")
            
            return filtered_accounts, filtered_amounts
        else:
            return [], []
    
    def _calculate_similarity(self, amounts):
        """计算金额相似度"""
        if not amounts or len(amounts) < 2:
            return 0
        
        max_amount = max(amounts)
        min_amount = min(amounts)
        
        if max_amount == 0:
            return 0
        
        return min_amount / max_amount
    
    def find_continuous_patterns(self, patterns):
        """查找连续对刷模式"""
        if not patterns:
            return []
        
        # 按账户组和对立类型分组
        pattern_groups = defaultdict(list)
        
        for pattern in patterns:
            # 创建分组键
            if pattern['检测类型'] == '单账户对刷':
                key = (tuple(pattern['账户组']), pattern['对立类型'], '单账户')
            else:
                key = (tuple(sorted(pattern['账户组'])), pattern['对立类型'], '多账户')
            
            pattern_groups[key].append(pattern)
        
        continuous_patterns = []
        
        for key, group_patterns in pattern_groups.items():
            # 按期号排序
            sorted_patterns = sorted(group_patterns, key=lambda x: x['局号'])
            
            # 检查是否连续
            if len(sorted_patterns) >= self.config.min_continuous_periods:
                # 计算统计信息
                total_amount = sum(p['总金额'] for p in sorted_patterns)
                avg_similarity = np.mean([p['相似度'] for p in sorted_patterns])
                
                # 获取游戏类型（取第一个模式的）
                game_type = sorted_patterns[0]['游戏类型']
                
                # 获取账户活跃度
                account_group = list(key[0])
                activity_level = self.get_account_group_activity_level(account_group, game_type)
                
                continuous_pattern = {
                    '账户组': account_group,
                    '游戏类型': game_type,
                    '账户数量': len(account_group),
                    '对立类型': key[1],
                    '检测类型': key[2],
                    '对刷局数': len(sorted_patterns),
                    '总投注金额': total_amount,
                    '平均相似度': avg_similarity,
                    '详细记录': sorted_patterns,
                    '账户活跃度': activity_level,
                    '要求最小对刷局数': self.get_required_min_periods(account_group, game_type)
                }
                
                continuous_patterns.append(continuous_pattern)
        
        return continuous_patterns
    
    def get_account_activity_level(self, account, game_type):
        """获取账户活跃度水平"""
        if game_type not in self.account_total_periods_by_game:
            return 'unknown'
        
        total_periods_stats = self.account_total_periods_by_game[game_type]
        periods = total_periods_stats.get(account, 0)
        
        return self._calculate_activity_level(periods)
    
    def get_account_group_activity_level(self, account_group, game_type):
        """获取账户组活跃度水平"""
        if game_type not in self.account_total_periods_by_game:
            return 'unknown'
        
        total_periods_stats = self.account_total_periods_by_game[game_type]
        
        min_periods = float('inf')
        for account in account_group:
            periods = total_periods_stats.get(account, 0)
            if periods < min_periods:
                min_periods = periods
        
        if min_periods == float('inf'):
            return 'unknown'
        
        return self._calculate_activity_level(min_periods)
    
    def _calculate_activity_level(self, periods):
        """根据局数计算活跃度水平"""
        if periods <= self.config.period_thresholds['low_activity']:
            return 'low'
        elif periods <= self.config.period_thresholds['medium_activity_high']:
            return 'medium'
        elif periods <= self.config.period_thresholds['high_activity_low']:
            return 'high'
        else:
            return 'very_high'
    
    def get_required_min_periods(self, account_group, game_type):
        """根据活跃度获取所需的最小对刷局数"""
        activity_level = self.get_account_group_activity_level(account_group, game_type)
        
        if activity_level == 'low':
            return self.config.period_thresholds['min_periods_low']
        elif activity_level == 'medium':
            return self.config.period_thresholds['min_periods_medium']
        elif activity_level == 'high':
            return self.config.period_thresholds['min_periods_high']
        else:
            return self.config.period_thresholds['min_periods_very_high']
    
    def display_detailed_results(self, patterns):
        """显示详细检测结果"""
        if not patterns:
            st.warning("⚠️ 未发现符合阈值条件的对刷行为")
            return
        
        # ========== 总体统计 ==========
        st.subheader("📊 总体统计")
        
        # 计算基础统计
        total_groups = len(patterns)
        total_accounts = sum(p['账户数量'] for p in patterns)
        total_wash_periods = sum(p['对刷局数'] for p in patterns)
        total_amount = sum(p['总投注金额'] for p in patterns)
        
        # 统计检测类型
        detection_type_stats = defaultdict(int)
        game_type_stats = defaultdict(int)
        opposite_type_stats = defaultdict(int)
        
        for pattern in patterns:
            detection_type_stats[pattern['检测类型']] += 1
            game_type_stats[pattern['游戏类型']] += 1
            opposite_type_stats[pattern['对立类型']] += 1
        
        # 第一行：基础数据统计
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("总对刷组数", f"{total_groups}")
        
        with col2:
            st.metric("涉及账户数", f"{total_accounts}")
        
        with col3:
            st.metric("总对刷局数", f"{total_wash_periods}")
        
        with col4:
            st.metric("总涉及金额", f"¥{total_amount:,.2f}")
        
        # ========== 检测类型统计 ==========
        st.subheader("🎯 检测类型统计")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**单账户对刷:**")
            single_count = detection_type_stats.get('单账户', 0)
            st.metric("检测组数", f"{single_count}组")
        
        with col2:
            st.write("**多账户对刷:**")
            multi_count = detection_type_stats.get('多账户', 0)
            st.metric("检测组数", f"{multi_count}组")
        
        # ========== 游戏类型统计 ==========
        st.subheader("🎲 游戏类型统计")
        
        if game_type_stats:
            cols = st.columns(min(3, len(game_type_stats)))
            for i, (game_type, count) in enumerate(game_type_stats.items()):
                if i < len(cols):
                    with cols[i]:
                        display_name = game_type
                        st.metric(
                            label=display_name,
                            value=f"{count}组"
                        )
        
        # ========== 对立类型统计 ==========
        st.subheader("⚔️ 对立类型统计")
        
        top_opposites = sorted(opposite_type_stats.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for opposite_type, count in top_opposites:
            if '-' in opposite_type:
                parts = opposite_type.split('-')
                display_type = f"{parts[0]} vs {parts[1]}"
            else:
                display_type = opposite_type
            
            st.write(f"**{display_type}**: {count}组")
        
        # ========== 详细对刷组分析 ==========
        st.subheader("🔍 详细对刷组分析")
        
        # 按游戏类型分组显示
        patterns_by_game = defaultdict(list)
        for pattern in patterns:
            patterns_by_game[pattern['游戏类型']].append(pattern)
        
        for game_type, game_patterns in patterns_by_game.items():
            game_icon = "🎰" if game_type == '百家乐' else "🐉" if game_type == '龙虎' else "🎲"
            
            with st.expander(f"{game_icon} {game_type}（发现{len(game_patterns)}组）", expanded=True):
                for i, pattern in enumerate(game_patterns, 1):
                    self._display_single_pattern(pattern, i, game_type)
    
    def _display_single_pattern(self, pattern, index, game_type):
        """显示单个对刷组详情"""
        st.markdown(f"**对刷组 {index}:**")
        
        # 账户信息
        if pattern['检测类型'] == '单账户对刷':
            st.markdown(f"**账户:** {pattern['账户组'][0]} (单账户对立下注)")
        else:
            st.markdown(f"**账户组:** {' ↔ '.join(pattern['账户组'])}")
        
        # 活跃度显示
        activity_icon = {
            'low': '🟢',
            'medium': '🟡', 
            'high': '🟠',
            'very_high': '🔴',
            'unknown': '⚪'
        }.get(pattern['账户活跃度'], '⚪')
        
        activity_text = {
            'low': '低活跃度',
            'medium': '中活跃度',
            'high': '高活跃度',
            'very_high': '极高活跃度',
            'unknown': '未知活跃度'
        }.get(pattern['账户活跃度'], '未知活跃度')
        
        st.markdown(f"**活跃度:** {activity_icon} {activity_text} | **游戏类型:** {game_type}")
        
        # 对立类型
        if '-' in pattern['对立类型']:
            parts = pattern['对立类型'].split('-')
            display_opposite = f"{parts[0]} vs {parts[1]}"
        else:
            display_opposite = pattern['对立类型']
        
        st.markdown(f"**对立类型:** {display_opposite} | **检测类型:** {pattern['检测类型']}")
        
        # 对刷统计
        st.markdown(f"**对刷局数:** {pattern['对刷局数']}局 (要求≥{pattern['要求最小对刷局数']}局)")
        st.markdown(f"**总投注金额:** ¥{pattern['总投注金额']:,.2f}")
        
        if pattern['检测类型'] == '多账户对刷':
            st.markdown(f"**平均相似度:** {pattern['平均相似度']:.2%}")
        
        # 详细记录
        st.markdown("**详细记录:**")
        
        for i, record in enumerate(pattern['详细记录'], 1):
            if pattern['检测类型'] == '单账户对刷':
                st.write(f"{i}. 局号: {record['局号']} | 下注: {record['下注玩法组'][0]}(¥{record['金额组'][0]:,.2f}) vs {record['下注玩法组'][1]}(¥{record['金额组'][1]:,.2f}) | 相似度: {record['相似度']:.2%}")
            else:
                accounts_info = []
                for acc, bet, amount in zip(record['账户组'], record['下注玩法组'], record['金额组']):
                    accounts_info.append(f"{acc}({bet}:¥{amount:,.2f})")
                
                st.write(f"{i}. 局号: {record['局号']} | 账户: {' ↔ '.join(accounts_info)} | 相似度: {record['相似度']:.2%}")
        
        if index < len(pattern):
            st.markdown("---")
    
    def export_detection_results(self, patterns, export_format='excel'):
        """导出检测结果"""
        if not patterns:
            st.warning("❌ 没有检测结果可供导出")
            return None
        
        try:
            main_data = []
            detailed_data = []
            
            for i, pattern in enumerate(patterns, 1):
                main_record = {
                    '组ID': f"组{i}",
                    '账户组': ' ↔ '.join(pattern['账户组']),
                    '游戏类型': pattern['游戏类型'],
                    '检测类型': pattern['检测类型'],
                    '对立类型': pattern['对立类型'],
                    '账户数量': pattern['账户数量'],
                    '对刷局数': pattern['对刷局数'],
                    '要求最小对刷局数': pattern['要求最小对刷局数'],
                    '总投注金额': pattern['总投注金额'],
                    '平均相似度': pattern['平均相似度'],
                    '账户活跃度': pattern['账户活跃度']
                }
                main_data.append(main_record)
                
                for j, record in enumerate(pattern['详细记录'], 1):
                    if pattern['检测类型'] == '单账户对刷':
                        detailed_record = {
                            '组ID': f"组{i}",
                            '账户组': ' ↔ '.join(pattern['账户组']),
                            '局号': record['局号'],
                            '游戏类型': record['游戏类型'],
                            '检测类型': record['检测类型'],
                            '下注玩法组': f"{record['下注玩法组'][0]} vs {record['下注玩法组'][1]}",
                            '金额组': f"¥{record['金额组'][0]:,.2f} vs ¥{record['金额组'][1]:,.2f}",
                            '总金额': record['总金额'],
                            '相似度': record['相似度']
                        }
                    else:
                        detailed_record = {
                            '组ID': f"组{i}",
                            '账户组': ' ↔ '.join(pattern['账户组']),
                            '局号': record['局号'],
                            '游戏类型': record['游戏类型'],
                            '检测类型': record['检测类型'],
                            '下注玩法组': ' ↔ '.join(record['下注玩法组']),
                            '金额组': ' ↔ '.join([f"¥{amt:,.2f}" for amt in record['金额组']]),
                            '总金额': record['总金额'],
                            '相似度': record['相似度']
                        }
                    detailed_data.append(detailed_record)
            
            df_main = pd.DataFrame(main_data)
            df_detailed = pd.DataFrame(detailed_data)
            
            # 格式化金额和百分比
            df_main['总投注金额'] = df_main['总投注金额'].apply(lambda x: f"¥{x:,.2f}")
            df_main['平均相似度'] = df_main['平均相似度'].apply(lambda x: f"{x:.2%}")
            
            df_detailed['总金额'] = df_detailed['总金额'].apply(lambda x: f"¥{x:,.2f}")
            df_detailed['相似度'] = df_detailed['相似度'].apply(lambda x: f"{x:.2%}")
            
            if export_format == 'excel':
                return self._export_to_excel(df_main, df_detailed)
            else:
                return self._export_to_csv(df_main, df_detailed)
                
        except Exception as e:
            logger.error(f"导出失败: {str(e)}")
            st.error(f"导出失败: {str(e)}")
            traceback.print_exc()
            return None
    
    def _export_to_excel(self, df_main, df_detailed):
        """导出到Excel格式"""
        try:
            output = io.BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_main.to_excel(writer, sheet_name='对刷组汇总', index=False)
                df_detailed.to_excel(writer, sheet_name='详细记录', index=False)
                
                workbook = writer.book
                main_sheet = workbook['对刷组汇总']
                detailed_sheet = workbook['详细记录']
                
                # 调整列宽
                for sheet in [main_sheet, detailed_sheet]:
                    for column in sheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        adjusted_width = min(max_length + 2, 50)
                        sheet.column_dimensions[column_letter].width = adjusted_width
                
                # 添加标题
                main_sheet.insert_rows(0, 3)
                main_sheet['A1'] = "百家乐对刷检测报告"
                main_sheet['A2'] = f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                main_sheet['A3'] = f"总对刷组数: {len(df_main)}"
                
                main_sheet.merge_cells('A1:L1')
                main_sheet.merge_cells('A2:L2')
                main_sheet.merge_cells('A3:L3')
                
                for cell in ['A1', 'A2', 'A3']:
                    main_sheet[cell].font = Font(bold=True, size=12)
                    main_sheet[cell].alignment = Alignment(horizontal='center')
            
            output.seek(0)
            return output
            
        except Exception as e:
            logger.error(f"Excel导出失败: {str(e)}")
            raise e
    
    def _export_to_csv(self, df_main, df_detailed):
        """导出到CSV格式"""
        try:
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                main_csv = df_main.to_csv(index=False, encoding='utf-8-sig')
                zip_file.writestr('对刷组汇总.csv', main_csv)
                
                detailed_csv = df_detailed.to_csv(index=False, encoding='utf-8-sig')
                zip_file.writestr('详细记录.csv', detailed_csv)
                
                readme_content = f"""百家乐对刷检测结果导出文件
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
总对刷组数: {len(df_main)}

文件说明:
1. 对刷组汇总.csv - 包含所有对刷组的汇总信息
2. 详细记录.csv - 包含每个对刷组的详细局号记录

检测参数:
- 最小投注金额: {self.config.min_amount}元
- 金额相似度阈值: {self.config.amount_similarity_threshold:.0%}
- 最大检测账户数: {self.config.max_accounts_in_group}
- 最小连续局数: {self.config.min_continuous_periods}
"""
                zip_file.writestr('说明.txt', readme_content)
            
            zip_buffer.seek(0)
            return zip_buffer
            
        except Exception as e:
            logger.error(f"CSV导出失败: {str(e)}")
            raise e
    
    def display_export_buttons(self, patterns):
        """显示导出按钮"""
        if not patterns:
            return
        
        st.markdown("---")
        st.subheader("📤 导出检测结果")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 导出Excel报告", use_container_width=True, type="primary"):
                with st.spinner("正在生成Excel报告..."):
                    excel_data = self.export_detection_results(patterns, 'excel')
                    if excel_data:
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        st.download_button(
                            label="⬇️ 下载Excel文件",
                            data=excel_data,
                            file_name=f"百家乐对刷检测报告_{timestamp}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
        
        with col2:
            if st.button("📄 导出CSV文件", use_container_width=True):
                with st.spinner("正在生成CSV文件..."):
                    csv_data = self.export_detection_results(patterns, 'csv')
                    if csv_data:
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        st.download_button(
                            label="⬇️ 下载CSV压缩包",
                            data=csv_data,
                            file_name=f"百家乐对刷检测报告_{timestamp}.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
        
        st.info(f"📊 导出内容: {len(patterns)}个对刷组, 共{sum(len(p['详细记录']) for p in patterns)}条详细记录")

# ==================== 主函数 ====================
def main():
    """主函数"""
    st.title("🎰 百家乐对刷检测系统")
    st.markdown("---")
    
    with st.sidebar:
        st.header("📁 数据上传")
        uploaded_file = st.file_uploader(
            "上传百家乐投注数据文件", 
            type=['xlsx', 'xls', 'csv'],
            help="请上传包含百家乐/龙虎投注数据的Excel或CSV文件"
        )
        
        st.header("⚙️ 检测参数设置")
        
        min_amount = st.slider(
            "最小投注金额阈值", 
            min_value=1, 
            max_value=100, 
            value=10,
            step=1,
            help="投注金额低于此值的记录将不参与检测"
        )
        
        max_accounts = st.slider(
            "最大检测账户数", 
            2, 8, 5, 
            help="检测的最大账户组合数量"
        )
        
        similarity_threshold = st.slider(
            "金额相似度阈值", 
            min_value=0.5, 
            max_value=1.0, 
            value=0.8,
            step=0.01,
            help="对刷金额的相似度要求（0.8表示80%相似）"
        )
        
        min_continuous_periods = st.slider(
            "最小连续局数", 
            min_value=1, 
            max_value=10, 
            value=3,
            help="要求的最小连续对刷局数"
        )
        
        st.subheader("💰 金额平衡设置")
        
        enable_balance_filter = st.checkbox("启用金额平衡过滤", value=True,
                                          help="确保对刷组内账户金额差距不超过设定倍数")
        
        max_ratio = 10
        if enable_balance_filter:
            max_ratio = st.slider("最大金额差距倍数", 
                                 min_value=2, 
                                 max_value=20, 
                                 value=10, 
                                 step=1,
                                 help="组内最大金额与最小金额的允许倍数")
        
        st.subheader("🎯 多账户匹配度配置")
        
        col1, col2 = st.columns(2)
        
        with col1:
            similarity_2_accounts = st.slider(
                "2个账户", 
                min_value=0.5, max_value=1.0, value=0.8, step=0.01,
                help="2个账户对刷的金额匹配度阈值"
            )
        
        with col2:
            similarity_3_accounts = st.slider(
                "3个账户", 
                min_value=0.5, max_value=1.0, value=0.85, step=0.01,
                help="3个账户对刷的金额匹配度阈值"
            )
        
        col1, col2 = st.columns(2)
        
        with col1:
            similarity_4_accounts = st.slider(
                "4个账户", 
                min_value=0.5, max_value=1.0, value=0.9, step=0.01,
                help="4个账户对刷的金额匹配度阈值"
            )
        
        with col2:
            similarity_5_accounts = st.slider(
                "5个账户", 
                min_value=0.5, max_value=1.0, value=0.95, step=0.01,
                help="5个账户对刷的金额匹配度阈值"
            )
    
    if uploaded_file is not None:
        try:
            # 创建配置
            config = BaccaratConfig()
            config.min_amount = min_amount
            config.max_accounts_in_group = max_accounts
            config.amount_similarity_threshold = similarity_threshold
            config.min_continuous_periods = min_continuous_periods
            
            config.amount_threshold = {
                'max_amount_ratio': max_ratio,
                'enable_threshold_filter': enable_balance_filter
            }
            
            config.account_count_similarity_thresholds = {
                2: similarity_2_accounts,
                3: similarity_3_accounts,
                4: similarity_4_accounts,
                5: similarity_5_accounts
            }
            
            # 创建检测器
            detector = BaccaratWashTradeDetector(config)
            
            st.success(f"✅ 已上传文件: {uploaded_file.name}")
            
            with st.spinner("🔄 正在解析数据..."):
                df_enhanced, filename = detector.upload_and_process(uploaded_file)
                
                if df_enhanced is not None and len(df_enhanced) > 0:
                    st.success(f"✅ 数据解析成功: {len(df_enhanced)} 条有效记录")
                    
                    with st.spinner("🔍 正在检测对刷交易..."):
                        patterns = detector.detect_all_wash_trades()
                    
                    if patterns:
                        st.success(f"✅ 检测完成: 发现 {len(patterns)} 个对刷模式")
                        detector.display_detailed_results(patterns)
                        detector.display_export_buttons(patterns)
                    else:
                        st.warning("⚠️ 未发现符合阈值条件的对刷行为")
                else:
                    st.error("❌ 数据解析失败，请检查文件格式和内容")
            
        except Exception as e:
            st.error(f"❌ 程序执行失败: {str(e)}")
            traceback.print_exc()
    else:
        st.info("👈 请在左侧边栏上传数据文件开始分析")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🎯 核心功能")
            st.markdown("""
            - **单账户对刷检测**
              - 同一账户下注庄和闲
              - 同一账户下注龙和虎
              - 自动识别各种变异体
            """)
        
        with col2:
            st.subheader("🔍 多账户检测")
            st.markdown("""
            - **多账户协作对刷**
              - 账户A下庄，账户B下闲
              - 账户A下龙，账户B下虎
              - 智能金额匹配分析
            """)
        
        with col3:
            st.subheader("📊 智能分析")
            st.markdown("""
            - **智能列名识别**
              - 支持各种列名变异体
              - 自动数据清洗
            - **专业报告导出**
              - Excel格式报告
              - CSV格式数据
            """)
    
    with st.expander("📖 系统使用说明", expanded=False):
        st.markdown("""
        ### 系统功能说明

        **🎯 检测逻辑：**
        1. **单账户对刷检测**
           - 检测同一账户在同一局中同时下注对立的玩法
           - 例如：同一账户同时下注"庄"和"闲"
           - 例如：同一账户同时下注"龙"和"虎"
        
        2. **多账户对刷检测**
           - 检测多个账户在同一局中分别下注对立的玩法
           - 例如：账户A下注"庄"，账户B下注"闲"
           - 例如：账户A下注"龙"，账户B下注"虎"
        
        **⚙️ 参数说明：**
        - **最小投注金额**：过滤小金额投注，避免误报
        - **金额相似度**：对刷金额的相似程度要求
        - **最小连续局数**：要求连续多局出现相同模式
        - **金额平衡过滤**：确保对刷组内金额比例合理
        
        **📊 数据要求：**
        - 文件格式：Excel(.xlsx/.xls) 或 CSV
        - 必要列：会员账号、局号、游戏类型、下注玩法、下注额度
        - 支持各种列名变异体（系统自动识别）
        
        **🎲 支持的游戏：**
        - 百家乐（Baccarat）各种变体
        - 龙虎斗（Dragon Tiger）各种变体
        
        **⚔️ 检测的对立关系：**
        - 庄 vs 闲
        - 龙 vs 虎
        - 庄 vs 和
        - 闲 vs 和
        - 庄对 vs 闲对
        - 大 vs 小
        - 单 vs 双
        
        **📈 输出结果：**
        - 详细的对刷组分析
        - 账户活跃度评级
        - 对刷金额统计
        - 一键导出报告
        """)

if __name__ == "__main__":
    main()
