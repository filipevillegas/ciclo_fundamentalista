import subprocess
import sys
import pandas as pd
import numpy as np
import warnings
import gspread
from google.auth import default
from google.auth.exceptions import DefaultCredentialsError
from gspread.exceptions import SpreadsheetNotFound, WorksheetNotFound
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import logging
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import importlib
import importlib.util
import gc
import hashlib
import pickle
import os

COLAB_AUTH = None
_colab_spec = importlib.util.find_spec("google.colab")
if _colab_spec:
    from google.colab import auth as _colab_auth  # type: ignore
    COLAB_AUTH = _colab_auth

# =============================================================================
# CONFIGURAÇÃO CENTRALIZADA
# =============================================================================
@dataclass
class Config:
    """Configuração centralizada de parâmetros do sistema"""

    # Pesos SCCS - ajustados para dados disponíveis
    SCCS_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        'SALES_GROWTH': 0.30,        # Aumentado
        'RETURN_COM_EQY': 0.25,      # Mantido
        'OPER_MARGIN': 0.20,         # Mantido
        'CF_FREE_CASH_FLOW': 0.15,   # Mantido
        'OPERATING_ROIC': 0.10,      # Ajustado
    })

    # Thresholds para scoring
    THRESHOLDS: Dict[str, Dict] = field(default_factory=lambda: {
        'SALES_GROWTH': {'neutral': 5, 'worst': -15, 'best': 30},
        'RETURN_COM_EQY': {'neutral': 12.5, 'scale': 10},
        'OPER_MARGIN': {'neutral': 10, 'worst': -5, 'best': 25},
        'OPERATING_ROIC': {'neutral': 8, 'worst': -5, 'best': 35},
        'CF_FREE_CASH_FLOW': {'neutral': 5, 'worst': -5, 'best': 15}
    })

    # Configurações de visualização
    PLOT_CONFIG: Dict = field(default_factory=lambda: {
        'figure_size': (14, 9),
        'lookback_periods': 16,
        'min_data_points': 3,
        'dpi': 100
    })

    # Configurações de validação
    VALIDATION: Dict = field(default_factory=lambda: {
        'max_gap_days': 120,
        'outlier_std_threshold': 3,
        'min_data_points': 8,
        'lookback_quarters': 4
    })

    # URLs e credenciais
    SHEET_URL: Optional[str] = None
    LOG_LEVEL: str = 'INFO'
    OUTPUT_FORMAT: str = 'csv'  # csv, excel, parquet
    GOOGLE_SERVICE_ACCOUNT_FILE: Optional[str] = None
    GOOGLE_WORKSHEET_NAME: Optional[str] = None

# =============================================================================
# SISTEMA DE LOGGING FINANCEIRO
# =============================================================================
class FinancialLogger:
    """Sistema de logging para auditoria de cálculos financeiros"""

    def __init__(self, config: Config):
        self.config = config
        log_filename = f'sccs_audit_{datetime.now():%Y%m%d_%H%M%S}.log'

        logging.basicConfig(
            level=getattr(logging, config.LOG_LEVEL),
            format='%(asctime)s | %(levelname)s | %(funcName)s | %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.audit_trail = []

    def log_calculation(self, empresa: str, metric: str, value: float, details: Dict = None):
        """Registra cálculo para auditoria"""
        msg = f"[{empresa}] {metric}: {value:.4f}"
        if details:
            msg += f" | Details: {json.dumps(details, default=str)}"
        self.logger.info(msg)

        self.audit_trail.append({
            'timestamp': datetime.now().isoformat(),
            'empresa': empresa,
            'metric': metric,
            'value': value,
            'details': details
        })

    def log_anomaly(self, empresa: str, issue: str, data: Any):
        """Registra anomalias detectadas"""
        self.logger.warning(f"[{empresa}] ANOMALY: {issue} | Data: {data}")

    def log_error(self, context: str, error: Exception):
        """Registra erros com contexto"""
        self.logger.error(f"ERROR in {context}: {str(error)}", exc_info=True)

    def export_audit_trail(self, filename: str = 'audit_trail.json'):
        """Exporta trilha de auditoria"""
        with open(filename, 'w') as f:
            json.dump(self.audit_trail, f, indent=2, default=str)
        self.logger.info(f"Audit trail exported to {filename}")

# =============================================================================
# INSTALADOR INTELIGENTE DE PACOTES
# =============================================================================
class PackageManager:
    """Gerenciador inteligente de pacotes com cache"""

    REQUIRED_PACKAGES = {
        'core': [
            'gspread', 'google-auth', 'google-colab',
            'pandas', 'numpy', 'matplotlib', 'seaborn', 'scipy'
        ],
        'financial': [
            'yfinance',           # Dados de mercado
            'pandas-ta',          # Indicadores técnicos
            'quantstats',         # Relatórios profissionais
        ],
        'visualization': [
            'plotly',            # Gráficos interativos
            'kaleido'           # Export de gráficos
        ],
        'performance': [
            'numba',            # Compilação JIT
            'joblib'            # Paralelização
        ]
    }

    @classmethod
    def install_packages(cls, categories: List[str] = None):
        """Instalação inteligente com verificação de cache"""
        if categories is None:
            categories = ['core', 'financial']

        for category in categories:
            packages = cls.REQUIRED_PACKAGES.get(category, [])
            for package in packages:
                cls._install_if_needed(package)

    @staticmethod
    def _install_if_needed(package: str):
        """Instala pacote apenas se necessário"""
        try:
            module_name = package.replace('-', '_')
            importlib.import_module(module_name)
            print(f"✓ {package} já instalado")
        except ImportError:
            print(f"📦 Instalando {package}...")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "-q", package
                ])
                print(f"✓ {package} instalado com sucesso")
            except Exception as e:
                print(f"⚠️ Erro ao instalar {package}: {e}")

# =============================================================================
# VALIDADOR DE DADOS FINANCEIROS
# =============================================================================
class DataValidator:
    """Validação e limpeza de dados para evitar vieses"""

    def __init__(self, config: Config, logger: FinancialLogger):
        self.config = config
        self.logger = logger

    def validate_temporal_consistency(self, df: pd.DataFrame) -> List[Dict]:
        """Verifica consistência temporal dos dados"""
        issues = []

        for empresa in df['EMPRESA'].unique():
            empresa_df = df[df['EMPRESA'] == empresa].sort_values('DATA')
            date_diff = empresa_df['DATA'].diff()

            # Detectar gaps temporais
            large_gaps = date_diff[date_diff > pd.Timedelta(days=self.config.VALIDATION['max_gap_days'])]
            if not large_gaps.empty:
                issue = {
                    'empresa': empresa,
                    'tipo': 'gap_temporal',
                    'datas': large_gaps.index.tolist(),
                    'gaps_days': large_gaps.dt.days.tolist()
                }
                issues.append(issue)
                self.logger.log_anomaly(empresa, 'Temporal gap detected', issue)

        return issues

    def detect_outliers(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """Detecta e marca outliers usando Z-score"""
        if columns is None:
            columns = list(self.config.SCCS_WEIGHTS.keys())

        outlier_mask = pd.DataFrame(False, index=df.index, columns=columns)

        for col in columns:
            if col in df.columns:
                std = df[col].std()
                if pd.isna(std) or np.isclose(std, 0):
                    outlier_mask[col] = False
                    continue

                mean = df[col].mean()
                z_scores = np.abs((df[col] - mean) / std)
                outlier_mask[col] = z_scores > self.config.VALIDATION['outlier_std_threshold']

                n_outliers = outlier_mask[col].sum()
                if n_outliers > 0:
                    self.logger.log_anomaly('GLOBAL', f'{n_outliers} outliers in {col}', None)

        return outlier_mask

    def prevent_look_ahead_bias(self, df: pd.DataFrame) -> pd.DataFrame:
        """Previne look-ahead bias validando dados temporais"""
        df = df.copy()

        for empresa in df['EMPRESA'].unique():
            mask = df['EMPRESA'] == empresa
            empresa_df = df[mask].sort_values('DATA')

            # Verificar consistência temporal dos dados
            for i in range(1, len(empresa_df)):
                current_date = empresa_df.iloc[i]['DATA']
                previous_date = empresa_df.iloc[i-1]['DATA']

                # Verificar se há gaps muito grandes entre dados consecutivos
                days_diff = (current_date - previous_date).days

                # Se o gap for muito grande (> 6 meses), marcar como suspeito
                if days_diff > 180:
                    self.logger.log_anomaly(
                        empresa,
                        f'Large temporal gap detected',
                        {'current': current_date, 'previous': previous_date, 'days': days_diff}
                    )
                    # Opcionalmente, marcar indicadores de crescimento como NaN
                    # se o gap for muito grande
                    if 'SALES_GROWTH' in df.columns and days_diff > 365:
                        df.loc[empresa_df.index[i], 'SALES_GROWTH'] = np.nan

        return df

    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Relatório completo de qualidade dos dados"""
        report = {
            'total_records': len(df),
            'unique_companies': df['EMPRESA'].nunique(),
            'date_range': {
                'start': df['DATA'].min(),
                'end': df['DATA'].max()
            },
            'missing_data': {},
            'outliers': {},
            'temporal_issues': []
        }

        # Análise de dados faltantes
        for col in self.config.SCCS_WEIGHTS.keys():
            if col in df.columns:
                missing_pct = (df[col].isna().sum() / len(df)) * 100
                report['missing_data'][col] = f"{missing_pct:.2f}%"

        # Detecção de outliers
        outlier_mask = self.detect_outliers(df)
        for col in outlier_mask.columns:
            report['outliers'][col] = outlier_mask[col].sum()

        # Issues temporais
        report['temporal_issues'] = self.validate_temporal_consistency(df)

        return report

# =============================================================================
# CALCULADOR SCCS OTIMIZADO
# =============================================================================
class OptimizedSCCSCalculator:
    """Calculador SCCS otimizado com vetorização e cache"""

    def __init__(self, config: Config, logger: FinancialLogger):
        self.config = config
        self.logger = logger
        self.cache = {}

    def get_cached_or_compute(self, key: str, compute_func, *args, **kwargs):
        """Cache com persistência em disco"""
        # Criar diretório de cache
        cache_dir = 'sccs_cache'
        os.makedirs(cache_dir, exist_ok=True)

        # Gerar hash único
        cache_key = hashlib.md5(
            f"{key}_{str(args)}_{str(kwargs)}".encode()
        ).hexdigest()

        cache_file = f"{cache_dir}/{cache_key}.pkl"

        # Verificar cache
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    self.logger.logger.info(f"Cache hit para {key}")
                    return pickle.load(f)
            except:
                pass  # Cache corrompido, recalcular

        # Calcular e salvar
        result = compute_func(*args, **kwargs)

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except:
            pass  # Falha ao salvar cache não é crítica

        return result

    def calculate_sccs_parallel(self, df: pd.DataFrame) -> np.ndarray:
        """Calcula SCCS em paralelo se possível"""
        try:
            from joblib import Parallel, delayed

            # Processar em paralelo por empresa
            def process_empresa(empresa_df):
                return self.calculate_sccs_vectorized(empresa_df)

            results = Parallel(n_jobs=-1, backend='threading')(
                delayed(process_empresa)(group)
                for _, group in df.groupby('EMPRESA')
            )

            return np.concatenate(results)

        except ImportError:
            # Fallback para versão serial
            self.logger.logger.info("joblib não disponível, usando processamento serial")
            return self.calculate_sccs_vectorized(df)

    def calculate_sccs_vectorized(self, df: pd.DataFrame) -> np.ndarray:
        """Calcula SCCS para todo DataFrame de uma vez usando vetorização"""
        scores_df = pd.DataFrame(index=df.index)

        # Calcular scores para cada indicador
        for indicator, weight in self.config.SCCS_WEIGHTS.items():
            if indicator in df.columns:
                threshold = self.config.THRESHOLDS.get(indicator, {})

                if indicator == 'RETURN_COM_EQY':
                    scores_df[indicator] = self._score_return_com_eqy_vectorized(df[indicator])
                elif indicator == 'CF_FREE_CASH_FLOW':
                    scores_df[indicator] = self._score_fcf_vectorized(df, indicator)
                else:
                    scores_df[indicator] = self._score_vectorized(
                        df[indicator].values,
                        threshold.get('neutral', 0),
                        threshold.get('worst', -10),
                        threshold.get('best', 10)
                    )

        # Calcular SCCS ponderado
        if scores_df.empty:
            self.logger.logger.warning(
                "Nenhuma coluna de indicador disponível para cálculo do SCCS; retornando zeros"
            )
            return np.zeros(len(df), dtype=float)

        weights = np.array([self.config.SCCS_WEIGHTS.get(col, 0) for col in scores_df.columns])
        weight_sum = weights.sum()

        if weight_sum <= 0 or not np.isfinite(weight_sum):
            normalized_weights = np.zeros_like(weights)
        else:
            normalized_weights = weights / weight_sum  # Normalizar pesos

        sccs_scores = (scores_df.values @ normalized_weights) * 5

        # Log de algumas métricas para auditoria
        for empresa in df['EMPRESA'].unique()[:5]:  # Log primeiras 5 empresas
            empresa_mask = df['EMPRESA'] == empresa
            if empresa_mask.any():
                idx = empresa_mask.idxmax()
                self.logger.log_calculation(
                    empresa,
                    'SCCS',
                    sccs_scores[idx],
                    {'scores': scores_df.loc[idx].to_dict()}
                )

        return sccs_scores

    @staticmethod
    def _score_vectorized(values: np.ndarray, neutral: float, worst: float, best: float) -> np.ndarray:
        """Função de scoring vetorizada"""
        scores = np.zeros_like(values, dtype=float)

        # Tratar NaN
        valid_mask = ~np.isnan(values)
        valid_values = values[valid_mask]

        # Aplicar scoring
        scores[valid_mask] = np.where(
            valid_values >= best, 10,
            np.where(
                valid_values <= worst, -10,
                np.where(
                    valid_values >= neutral,
                    10 * (valid_values - neutral) / (best - neutral),
                    -10 * (neutral - valid_values) / (neutral - worst)
                )
            )
        )

        return scores

    def _score_return_com_eqy_vectorized(self, values: pd.Series) -> np.ndarray:
        """Score ROE usando tangente hiperbólica"""
        threshold = self.config.THRESHOLDS['RETURN_COM_EQY']
        return np.clip(
            np.tanh((values - threshold['neutral']) / threshold['scale']) * 10,
            -10, 10
        )

    def _score_fcf_vectorized(self, df: pd.DataFrame, indicator: str) -> np.ndarray:
        """Score FCF com análise histórica - CORRIGIDO COM PROTEÇÃO CONTRA DIVISÃO POR ZERO"""
        scores = np.zeros(len(df))

        for empresa in df['EMPRESA'].unique():
            mask = df['EMPRESA'] == empresa
            empresa_df = df[mask]

            # Se REVENUE existir, usar para calcular margem FCF
            if 'REVENUE' in df.columns:
                # CORREÇÃO CRÍTICA: Proteção contra divisão por zero
                revenue_safe = empresa_df['REVENUE'].replace(0, np.nan)
                fcf_margin = (empresa_df[indicator] / revenue_safe) * 100

                scores[mask] = self._score_vectorized(
                    fcf_margin.values,
                    self.config.THRESHOLDS['CF_FREE_CASH_FLOW']['neutral'],
                    self.config.THRESHOLDS['CF_FREE_CASH_FLOW']['worst'],
                    self.config.THRESHOLDS['CF_FREE_CASH_FLOW']['best']
                )
            else:
                # Caso contrário, usar o valor absoluto do FCF
                scores[mask] = self._score_vectorized(
                    empresa_df[indicator].values,
                    self.config.THRESHOLDS['CF_FREE_CASH_FLOW']['neutral'],
                    self.config.THRESHOLDS['CF_FREE_CASH_FLOW']['worst'],
                    self.config.THRESHOLDS['CF_FREE_CASH_FLOW']['best']
                )

        return scores

# =============================================================================
# MÉTRICAS FINANCEIRAS AVANÇADAS
# =============================================================================
class FinancialMetrics:
    """Cálculo de métricas financeiras profissionais"""

    def __init__(self, config: Config, logger: FinancialLogger):
        self.config = config
        self.logger = logger

    def calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calcula métricas de risco padrão do mercado"""
        try:
            metrics = {
                'mean_return': returns.mean(),
                'volatility': returns.std(),
                'sharpe_ratio': self._sharpe_ratio(returns),
                'sortino_ratio': self._sortino_ratio(returns),
                'max_drawdown': self._max_drawdown(returns),
                'var_95': returns.quantile(0.05),
                'cvar_95': returns[returns <= returns.quantile(0.05)].mean(),
                'skewness': returns.skew(),
                'kurtosis': returns.kurtosis()
            }

            self.logger.log_calculation('PORTFOLIO', 'Risk Metrics', 0, metrics)
            return metrics

        except Exception as e:
            self.logger.log_error('calculate_risk_metrics', e)
            return {}

    @staticmethod
    def _sharpe_ratio(returns: pd.Series, risk_free: float = 0.02) -> float:
        """Calcula Sharpe Ratio"""
        excess_returns = returns - risk_free/252
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    @staticmethod
    def _sortino_ratio(returns: pd.Series, risk_free: float = 0.02) -> float:
        """Calcula Sortino Ratio"""
        excess_returns = returns - risk_free/252
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std()
        return np.sqrt(252) * excess_returns.mean() / downside_std if downside_std > 0 else 0

    @staticmethod
    def _max_drawdown(returns: pd.Series) -> float:
        """Calcula Maximum Drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def regime_detection(self, data: pd.DataFrame) -> List[str]:
        """Detecta mudanças de regime no mercado"""
        try:
            from sklearn.mixture import GaussianMixture

            # Preparar features
            features = data[['SCCS', 'DIRECAO', 'ACELERACAO']].fillna(0).values

            # Modelo de mistura gaussiana
            gmm = GaussianMixture(n_components=3, random_state=42)
            regimes = gmm.fit_predict(features)

            # Mapear regimes baseado nas médias
            means = gmm.means_[:, 0]  # Usar SCCS como referência
            sorted_idx = np.argsort(means)
            regime_map = {sorted_idx[0]: 'Bear', sorted_idx[1]: 'Neutral', sorted_idx[2]: 'Bull'}

            return [regime_map[r] for r in regimes]

        except ImportError:
            self.logger.log_anomaly('GLOBAL', 'sklearn not available for regime detection', None)
            return ['Unknown'] * len(data)

# =============================================================================
# NOVA FUNCIONALIDADE: PREVISÃO DE CICLOS
# =============================================================================
class CyclePrediction:
    """Sistema de previsão do próximo ciclo microeconômico"""

    def __init__(self, config: Config, logger: FinancialLogger):
        self.config = config
        self.logger = logger
        self.transition_matrix = None
        self.cycle_history = {}

    def build_transition_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Constrói matriz de transição de Markov"""
        cycles = ['Expansão', 'Retração', 'Desaceleração', 'Recuperação']
        matrix = np.zeros((4, 4))

        for empresa in df['EMPRESA'].unique():
            empresa_df = df[df['EMPRESA'] == empresa].sort_values('DATA')

            # Guardar histórico
            self.cycle_history[empresa] = empresa_df['CICLO'].tolist()

            # Calcular transições
            for i in range(len(empresa_df) - 1):
                current = empresa_df.iloc[i]['CICLO']
                next_cycle = empresa_df.iloc[i + 1]['CICLO']

                if current in cycles and next_cycle in cycles:
                    curr_idx = cycles.index(current)
                    next_idx = cycles.index(next_cycle)
                    matrix[curr_idx, next_idx] += 1

        # Normalizar
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, where=row_sums != 0)

        self.transition_matrix = matrix
        self.logger.log_calculation('SYSTEM', 'Transition Matrix Built', 0,
                                   {'shape': matrix.shape})
        return matrix

    def predict_next_cycle(self, empresa: str, current_data: pd.Series) -> Dict:
        """Prevê próximo ciclo com confiança"""
        cycles = ['Expansão', 'Retração', 'Desaceleração', 'Recuperação']
        current_cycle = current_data.get('CICLO', 'Indefinido')

        # Probabilidades base (Markov)
        if self.transition_matrix is not None and current_cycle in cycles:
            curr_idx = cycles.index(current_cycle)
            base_probs = self.transition_matrix[curr_idx]
        else:
            base_probs = np.ones(4) / 4

        # Ajuste por indicadores
        indicator_probs = self._analyze_indicators(current_data)

        # Ajuste por padrão histórico
        pattern_probs = self._analyze_pattern(empresa)

        # Combinação ponderada
        final_probs = (base_probs * 0.5 +
                      indicator_probs * 0.3 +
                      pattern_probs * 0.2)

        # Normalizar
        final_probs = final_probs / final_probs.sum()

        # Resultado
        predicted_idx = np.argmax(final_probs)

        return {
            'current_cycle': current_cycle,
            'predicted_cycle': cycles[predicted_idx],
            'confidence': float(final_probs[predicted_idx]),
            'probabilities': dict(zip(cycles, final_probs.tolist())),
            'indicators': {
                'SCCS': current_data.get('SCCS', 0),
                'SALES_GROWTH': current_data.get('SALES_GROWTH', 0),
                'ROE': current_data.get('RETURN_COM_EQY', 0),
                'MARGIN': current_data.get('OPER_MARGIN', 0)
            }
        }

    def _analyze_indicators(self, data: pd.Series) -> np.ndarray:
        """Análise baseada em indicadores"""
        probs = np.zeros(4)  # [Expansão, Retração, Desaceleração, Recuperação]

        sales = data.get('SALES_GROWTH', 0)
        roe = data.get('RETURN_COM_EQY', 0)
        margin = data.get('OPER_MARGIN', 0)
        sccs = data.get('SCCS', 0)

        # Lógica de probabilidades
        if sales > 20 and roe > 15:
            probs[0] = 0.6  # Expansão
        elif sales > 5 and sales < 15:
            probs[1] = 0.5  # Retração
        elif sales < 0 or roe < 5:
            probs[2] = 0.6  # Desaceleração
        elif sales > 0 and sccs > 0:
            probs[3] = 0.5  # Recuperação

        # Normalizar
        if probs.sum() == 0:
            probs = np.ones(4) / 4
        else:
            probs = probs / probs.sum()

        return probs

    def _analyze_pattern(self, empresa: str) -> np.ndarray:
        """Análise de padrões históricos"""
        if empresa not in self.cycle_history:
            return np.ones(4) / 4

        history = self.cycle_history[empresa]
        if len(history) < 3:
            return np.ones(4) / 4

        # Identificar padrões comuns
        recent = history[-3:]
        cycles = ['Expansão', 'Retração', 'Desaceleração', 'Recuperação']
        probs = np.zeros(4)

        # Padrões conhecidos
        if recent == ['Desaceleração', 'Desaceleração', 'Recuperação']:
            probs[0] = 0.6  # Provável Expansão
        elif recent == ['Expansão', 'Expansão', 'Retração']:
            probs[2] = 0.5  # Possível Desaceleração

        # Normalizar
        if probs.sum() == 0:
            probs = np.ones(4) / 4
        else:
            probs = probs / probs.sum()

        return probs

# =============================================================================
# VISUALIZADOR INTERATIVO
# =============================================================================
class InteractiveVisualizer:
    """Criação de visualizações interativas com Plotly"""

    def __init__(self, config: Config, logger: FinancialLogger):
        self.config = config
        self.logger = logger

    def create_interactive_dashboard(self, results_df: pd.DataFrame, metrics: Dict) -> Any:
        """Dashboard interativo para análise"""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'SCCS Evolution', 'Cycle Distribution',
                    'Performance Heatmap', 'Risk Metrics',
                    'Correlation Matrix', 'Regime Analysis'
                ),
                specs=[
                    [{'type': 'scatter'}, {'type': 'pie'}],
                    [{'type': 'heatmap'}, {'type': 'bar'}],
                    [{'type': 'heatmap'}, {'type': 'scatter'}]
                ],
                vertical_spacing=0.1,
                horizontal_spacing=0.15
            )

            # 1. SCCS Evolution
            for empresa in results_df['EMPRESA'].unique()[:10]:  # Top 10 empresas
                empresa_data = results_df[results_df['EMPRESA'] == empresa]

                fig.add_trace(
                    go.Scatter(
                        x=empresa_data['DATA'],
                        y=empresa_data['SCCS'],
                        name=empresa,
                        mode='lines+markers',
                        hovertemplate='<b>%{text}</b><br>SCCS: %{y:.2f}<br>Data: %{x}',
                        text=empresa_data['EMPRESA']
                    ),
                    row=1, col=1
                )

            # 2. Cycle Distribution
            cycle_counts = results_df['CICLO'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=cycle_counts.index,
                    values=cycle_counts.values,
                    hole=0.3,
                    marker=dict(colors=['#27ae60', '#c0392b', '#7f8c8d', '#2c3e50'])
                ),
                row=1, col=2
            )

            # 3. Performance Heatmap
            pivot_data = results_df.pivot_table(
                index='EMPRESA',
                columns='TRIMESTRE_STR',
                values='SCCS',
                aggfunc='mean'
            ).iloc[:15, -8:]  # Top 15 empresas, últimos 8 trimestres

            fig.add_trace(
                go.Heatmap(
                    z=pivot_data.values,
                    x=pivot_data.columns,
                    y=pivot_data.index,
                    colorscale='RdYlGn',
                    zmid=0
                ),
                row=2, col=1
            )

            # 4. Risk Metrics Bar Chart
            if metrics:
                fig.add_trace(
                    go.Bar(
                        x=list(metrics.keys()),
                        y=list(metrics.values()),
                        marker_color='lightblue'
                    ),
                    row=2, col=2
                )

            # 5. Correlation Matrix
            indicators = [col for col in results_df.columns if col in self.config.SCCS_WEIGHTS.keys()]
            if indicators:
                corr_matrix = results_df[indicators].corr()
                fig.add_trace(
                    go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.index,
                        colorscale='Viridis'
                    ),
                    row=3, col=1
                )

            # 6. Regime Analysis
            if 'REGIME' in results_df.columns:
                regime_data = results_df.groupby(['DATA', 'REGIME']).size().unstack(fill_value=0)
                for regime in regime_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=regime_data.index,
                            y=regime_data[regime],
                            name=regime,
                            stackgroup='one'
                        ),
                        row=3, col=2
                    )

            # Configurar layout
            fig.update_layout(
                height=1200,
                showlegend=True,
                hovermode='x unified',
                template='plotly_dark',
                title_text="SCCS Analysis Dashboard - Professional Edition",
                title_font_size=20
            )

            return fig

        except ImportError:
            self.logger.log_error('Plotly not available', Exception('Install plotly for interactive charts'))
            return None

    def create_cycle_plot_enhanced(self, empresa: str, results_df: pd.DataFrame):
        """Versão melhorada do gráfico de ciclo - COM CORREÇÃO DE MATPLOTLIB STYLE"""
        empresa_df = results_df[results_df['EMPRESA'] == empresa].tail(
            self.config.PLOT_CONFIG['lookback_periods']
        )

        if len(empresa_df) < self.config.PLOT_CONFIG['min_data_points']:
            self.logger.log_anomaly(empresa, 'Insufficient data for cycle plot', len(empresa_df))
            return

        # CORREÇÃO: Validar e aplicar style matplotlib
        available_styles = plt.style.available
        if 'seaborn-v0_8-darkgrid' in available_styles:
            plt.style.use('seaborn-v0_8-darkgrid')
        elif 'seaborn' in available_styles:
            plt.style.use('seaborn')

        fig, ax = plt.subplots(figsize=self.config.PLOT_CONFIG['figure_size'])

        # Configurar limites dinâmicos
        max_abs_x = empresa_df['DIRECAO'].abs().max()
        max_abs_y = empresa_df['ACELERACAO'].abs().max()
        plot_limit = max(max_abs_x, max_abs_y, 5) * 1.2

        ax.set_xlim(-plot_limit, plot_limit)
        ax.set_ylim(-plot_limit, plot_limit)

        # Áreas dos quadrantes com preenchimento sólido
        from matplotlib.patches import Rectangle
        from matplotlib.collections import PatchCollection

        # Configurar preenchimento para cada quadrante
        quadrants = {
            'Expansão': {'xy': (0, 0), 'color': '#27ae60', 'alpha': 0.3},
            'Retração': {'xy': (0, -plot_limit), 'color': '#c0392b', 'alpha': 0.3},
            'Desaceleração': {'xy': (-plot_limit, -plot_limit), 'color': '#7f8c8d', 'alpha': 0.3},
            'Recuperação': {'xy': (-plot_limit, 0), 'color': '#2c3e50', 'alpha': 0.3}
        }

        for quad_name, props in quadrants.items():
            rect = Rectangle(
                props['xy'],
                plot_limit if props['xy'][0] >= 0 else plot_limit,
                plot_limit if props['xy'][1] >= 0 else plot_limit,
                facecolor=props['color'],
                alpha=props['alpha'],
                label=quad_name
            )
            ax.add_patch(rect)

        # Spline suavizado com mais pontos
        x = empresa_df['DIRECAO'].values
        y = empresa_df['ACELERACAO'].values
        t = np.arange(len(x))

        if len(x) >= 4:  # Precisa de pelo menos 4 pontos para spline cúbico
            spline_x = CubicSpline(t, x)
            spline_y = CubicSpline(t, y)
            t_smooth = np.linspace(t.min(), t.max(), 500)
            x_smooth = spline_x(t_smooth)
            y_smooth = spline_y(t_smooth)

            # Linha com gradiente de cor
            points = np.array([x_smooth, y_smooth]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            from matplotlib.collections import LineCollection
            lc = LineCollection(segments, cmap='viridis', linewidth=3)
            lc.set_array(t_smooth)
            ax.add_collection(lc)
        else:
            ax.plot(x, y, color='#00aaff', linewidth=2.5)

        # Pontos com tamanhos variáveis baseados em SCCS
        sccs_normalized = (empresa_df['SCCS'] - empresa_df['SCCS'].min()) / (empresa_df['SCCS'].max() - empresa_df['SCCS'].min() + 0.001)
        sizes = 50 + sccs_normalized * 150

        scatter = ax.scatter(x, y, c=empresa_df['SCCS'], s=sizes,
                           cmap='RdYlGn', edgecolors='white', linewidth=2,
                           alpha=0.8, zorder=5)

        # Colorbar para SCCS
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label('SCCS Score', rotation=270, labelpad=15, color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

        # Anotações melhoradas
        for i, row in empresa_df.iterrows():
            # Apenas mostrar labels para pontos importantes
            if i == empresa_df.index[0] or i == empresa_df.index[-1] or abs(row['SCCS']) > empresa_df['SCCS'].quantile(0.9):
                ax.annotate(
                    row['TRIMESTRE_STR'],
                    xy=(row['DIRECAO'], row['ACELERACAO']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='purple', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', color='white', lw=0.5)
                )

        # Posição atual com destaque
        last_point = empresa_df.iloc[-1]
        ax.scatter(last_point['DIRECAO'], last_point['ACELERACAO'],
                  color='yellow', s=300, marker='*', edgecolors='red',
                  linewidth=2, zorder=10, label='Posição Atual')

        # Grid e estilo
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        ax.axhline(0, color='white', lw=1, linestyle='-', alpha=0.5)
        ax.axvline(0, color='white', lw=1, linestyle='-', alpha=0.5)

        # Títulos e labels
        ax.set_title(f'Ciclo Fundamentalista Avançado - {empresa}',
                    fontsize=18, color='white', fontweight='bold', pad=20)
        ax.set_xlabel('Direção (Nível dos Fundamentos)', fontsize=12, color='white')
        ax.set_ylabel('Aceleração (Momentum dos Fundamentos)', fontsize=12, color='white')

        # Estilo dark theme
        ax.set_facecolor('#0a0a0a')
        fig.set_facecolor('#0a0a0a')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
            spine.set_linewidth(0.5)

        # Labels dos quadrantes
        quadrant_labels = {
            'EXPANSÃO': (plot_limit*0.5, plot_limit*0.9),
            'RETRAÇÃO': (plot_limit*0.5, -plot_limit*0.9),
            'DESACELERAÇÃO': (-plot_limit*0.5, -plot_limit*0.9),
            'RECUPERAÇÃO': (-plot_limit*0.5, plot_limit*0.9)
        }

        for label, (x_pos, y_pos) in quadrant_labels.items():
            ax.text(x_pos, y_pos, label, ha='center', va='center',
                   fontsize=14, color='white', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.5))

        # Legenda
        ax.legend(loc='upper left', framealpha=0.9, facecolor='black',
                 edgecolor='white', labelcolor='white')

        # Informações adicionais
        info_text = f"SCCS Atual: {last_point['SCCS']:.2f}\n"
        info_text += f"Ciclo: {last_point['CICLO']}\n"
        info_text += f"Tendência: {'↑' if last_point['DIRECAO'] > 0 else '↓'}"

        ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
               fontsize=10, color='white', verticalalignment='bottom',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))

        plt.tight_layout()
        plt.show()

# =============================================================================
# PIPELINE PRINCIPAL REFATORADO
# =============================================================================
class SCCSAnalysisPipeline:
    """Pipeline principal com todas as melhorias implementadas"""

    def __init__(self, sheet_url: str, config: Config = None):
        self.config = config or Config()
        self.config.SHEET_URL = sheet_url
        self.logger = FinancialLogger(self.config)
        self.validator = DataValidator(self.config, self.logger)
        self.calculator = OptimizedSCCSCalculator(self.config, self.logger)
        self.metrics = FinancialMetrics(self.config, self.logger)
        self.visualizer = InteractiveVisualizer(self.config, self.logger)

        self.df = None
        self.results_df = pd.DataFrame()
        self.risk_metrics = {}
        self.dashboard = None
        self.predictor = None  # Para o sistema de previsão
        self.predictions = {}  # Para armazenar previsões

    def validate_sheet_structure(self, all_values):
        """Valida estrutura antes de processar"""
        if len(all_values) < 3:
            raise ValueError("Planilha precisa ter pelo menos 3 linhas")

        # Verificar formato esperado
        if not all_values[0] or not all_values[1]:
            raise ValueError("Headers inválidos")

        # Verificar se tem dados
        if len(all_values[2:]) == 0:
            raise ValueError("Planilha sem dados")

        return True

    def cleanup_memory(self):
        """Libera memória após processamento pesado"""
        # Limpar DataFrames temporários
        temp_attrs = ['_temp_df', '_cache_df', '_intermediate_results']
        for attr in temp_attrs:
            if hasattr(self, attr):
                delattr(self, attr)

        # Forçar garbage collection
        gc.collect()

        self.logger.logger.info("Memória liberada com sucesso")

    def _create_gspread_client(self):
        """Cria cliente gspread com múltiplas estratégias de autenticação"""
        service_file = self.config.GOOGLE_SERVICE_ACCOUNT_FILE
        if service_file:
            service_path = os.path.expanduser(service_file)
            if not os.path.exists(service_path):
                raise FileNotFoundError(
                    f"Arquivo de credenciais não encontrado: {service_path}"
                )
            return gspread.service_account(filename=service_path)

        env_credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if env_credentials:
            env_path = os.path.expanduser(env_credentials)
            if os.path.exists(env_path):
                return gspread.service_account(filename=env_path)

        try:
            if COLAB_AUTH is not None:
                COLAB_AUTH.authenticate_user()
                creds, _ = default()
                return gspread.authorize(creds)

            creds, _ = default()
            return gspread.authorize(creds)
        except DefaultCredentialsError as exc:
            raise RuntimeError(
                "Credenciais do Google não encontradas. Defina GOOGLE_SERVICE_ACCOUNT_FILE na Config ou a variável de ambiente GOOGLE_APPLICATION_CREDENTIALS."
            ) from exc

    def load_and_prepare_data(self) -> bool:
        """Carrega e prepara dados com validações completas - VERSÃO CORRIGIDA"""
        try:
            self.logger.logger.info("=== INICIANDO CARREGAMENTO DE DADOS ===")

            if not self.config.SHEET_URL:
                raise ValueError("URL da planilha não configurada na Config")

            # AUTENTICAÇÃO SIMPLIFICADA E CORRETA
            print("🔐 Autenticando com Google Sheets...")
            gc_client = self._create_gspread_client()
            print("✅ Autenticação concluída!")

            # Extrair sheet_id da URL
            if '/d/' in self.config.SHEET_URL:
                sheet_id = self.config.SHEET_URL.split('/d/')[1].split('/')[0]
            else:
                raise ValueError("URL da planilha inválida")

            print(f"📄 Acessando planilha ID: {sheet_id[:10]}...")

            # Abrir planilha
            spreadsheet = gc_client.open_by_key(sheet_id)
            worksheet_name = self.config.GOOGLE_WORKSHEET_NAME
            if worksheet_name:
                print(f"📄 Utilizando aba configurada: {worksheet_name}")
                worksheet = spreadsheet.worksheet(worksheet_name)
            else:
                worksheet = spreadsheet.sheet1
            all_values = worksheet.get_all_values()

            print("✅ Planilha acessada com sucesso!")

            # NOVA VALIDAÇÃO
            self.validate_sheet_structure(all_values)

            # Processar headers
            header_empresas = all_values[0]
            header_indicadores = all_values[1]
            data_rows = all_values[2:]

            print(f"📊 Processando {len(data_rows)} linhas de dados...")

            dados_processados = []
            coluna_map = {
                i: (e.strip(), ind.strip())
                for i, (e, ind) in enumerate(zip(header_empresas, header_indicadores))
                if e and ind
            }

            # Processar linhas
            for row_idx, row in enumerate(data_rows):
                data_str = row[0] if len(row) > 0 else None
                if not data_str:
                    continue

                dados_por_empresa = {}
                for col_idx, (empresa, indicador) in coluna_map.items():
                    if col_idx == 0:  # Pular coluna de data
                        continue

                    if empresa not in dados_por_empresa:
                        dados_por_empresa[empresa] = {
                            'EMPRESA': empresa,
                            'DATA': data_str
                        }

                    valor_str = row[col_idx] if col_idx < len(row) else ''
                    try:
                        cleaned_val = str(valor_str).replace('%', '').strip()
                        if cleaned_val in ('', '-', '#N/A', 'N/A', 'None'):
                            valor = np.nan
                        else:
                            valor = float(cleaned_val.replace('.', '').replace(',', '.'))
                    except (ValueError, TypeError) as e:
                        valor = np.nan
                        if row_idx < 5:  # Log apenas primeiras linhas
                            self.logger.log_anomaly(
                                empresa,
                                f"Error parsing {indicador}",
                                {'value': valor_str, 'error': str(e)}
                            )

                    dados_por_empresa[empresa][indicador] = valor

                dados_processados.extend(dados_por_empresa.values())

            # Criar DataFrame
            df = pd.DataFrame(dados_processados)
            if df.empty:
                raise ValueError("Nenhum dado processado")

            # Processar datas
            df['DATA'] = pd.to_datetime(df['DATA'], format='%d/%m/%Y', errors='coerce')
            df.dropna(subset=['DATA'], inplace=True)

            # Renomear colunas conforme necessário
            df.rename(columns={
                'NET_DEBT_TO_SHRHLDR_EQTY': 'NET_DEBT_EQTY',
                'CASH_FLOW_PER_SH': 'CF_FREE_CASH_FLOW',
                'OPERATING_ROIC': 'OPERATING_ROIC'
            }, inplace=True)

            # Garantir que todas as colunas necessárias existam
            for indicator in self.config.SCCS_WEIGHTS.keys():
                if indicator not in df.columns:
                    df[indicator] = np.nan
                    self.logger.logger.warning(f"Coluna {indicator} não encontrada, preenchida com NaN")

            # Ordenar por empresa e data
            df = df.sort_values(['EMPRESA', 'DATA']).reset_index(drop=True)

            # CORREÇÃO: SALES_GROWTH já vem da planilha, não precisa calcular
            if 'SALES_GROWTH' in df.columns:
                df['SALES_GROWTH'].replace([np.inf, -np.inf], np.nan, inplace=True)
            else:
                df['SALES_GROWTH'] = np.nan

            # Validações
            self.logger.logger.info("=== EXECUTANDO VALIDAÇÕES ===")

            # 1. Detectar outliers
            outliers = self.validator.detect_outliers(df)
            self.logger.logger.info(f"Outliers detectados: {outliers.sum().sum()}")

            # 2. Validar consistência temporal
            temporal_issues = self.validator.validate_temporal_consistency(df)
            self.logger.logger.info(f"Issues temporais: {len(temporal_issues)}")

            # 3. Prevenir look-ahead bias
            df = self.validator.prevent_look_ahead_bias(df)

            # 4. Relatório de qualidade
            quality_report = self.validator.validate_data_quality(df)
            self.logger.logger.info(f"Qualidade dos dados: {quality_report}")

            self.df = df
            self.logger.logger.info(
                f"✅ {len(df)} registros carregados de {df['EMPRESA'].nunique()} empresas"
            )

            return True

        except WorksheetNotFound as e:
            self.logger.log_error('load_and_prepare_data', e)
            print(f"\n❌ Aba configurada não encontrada: {self.config.GOOGLE_WORKSHEET_NAME}")
            print("Verifique o nome da aba na planilha do Google Sheets.")
            return False

        except SpreadsheetNotFound as e:
            self.logger.log_error('load_and_prepare_data', e)
            print("\n❌ Planilha não encontrada ou sem acesso.")
            print("Confirme o ID da planilha e as permissões de compartilhamento.")
            return False

        except Exception as e:
            self.logger.log_error('load_and_prepare_data', e)
            print(f"\n❌ Erro ao carregar dados: {str(e)}")

            # Mensagem de ajuda adicional
            if "403" in str(e) or "permission" in str(e).lower():
                print("\n" + "="*80)
                print("⚠️  ERRO: Acesso negado à planilha.")
                print("="*80)
                print("\nPossíveis soluções:")
                print("\n1. Verifique as permissões da planilha:")
                print("   - Abra a planilha no Google Sheets")
                print("   - Clique em 'Compartilhar' (botão verde no canto superior direito)")
                print("   - Configure para 'Qualquer pessoa com o link pode visualizar'")
                print("\n2. Se a planilha é privada:")
                print("   - Compartilhe com seu email do Google")
                print("   - Ou faça uma cópia para sua conta")
                print("="*80)

            return False

    def run_analysis(self):
        """Executa análise completa com SCCS, métricas e PREVISÕES"""
        if self.df is None or self.df.empty:
            self.logger.logger.error("Dados não carregados. Abortando análise.")
            return

        self.logger.logger.info("=== INICIANDO ANÁLISE SCCS ===")

        # Calcular SCCS vetorizado (tentar paralelo se disponível)
        self.df['SCCS'] = self.calculator.calculate_sccs_parallel(self.df)

        # Calcular dinâmica de ciclo
        self.calculate_cycle_dynamics()

        # NOVA FUNCIONALIDADE: Inicializar sistema de previsão
        self.predictor = CyclePrediction(self.config, self.logger)

        # Construir modelo de transição
        self.predictor.build_transition_matrix(self.results_df)

        # Fazer previsões para cada empresa
        self.predictions = {}
        for empresa in self.results_df['EMPRESA'].unique():
            empresa_data = self.results_df[self.results_df['EMPRESA'] == empresa]
            if not empresa_data.empty:
                latest = empresa_data.iloc[-1]
                prediction = self.predictor.predict_next_cycle(empresa, latest)
                self.predictions[empresa] = prediction

                # Log
                self.logger.log_calculation(
                    empresa,
                    'CYCLE_PREDICTION',
                    prediction['confidence'],
                    prediction
                )

        # Detectar regimes de mercado
        if len(self.results_df) > 0:
            self.results_df['REGIME'] = self.metrics.regime_detection(self.results_df)

        # Calcular métricas de risco
        if 'SCCS' in self.results_df.columns:
            # Calcular retornos do SCCS
            sccs_returns = self.results_df.groupby('EMPRESA')['SCCS'].pct_change()
            sccs_returns = sccs_returns.dropna()

            if len(sccs_returns) > 20:  # Precisa de dados suficientes
                self.risk_metrics = self.metrics.calculate_risk_metrics(sccs_returns)
                self.logger.logger.info(f"Métricas de risco calculadas: {self.risk_metrics}")

        # Limpar memória após processamento pesado
        self.cleanup_memory()

        self.logger.logger.info("✅ Análise SCCS, métricas avançadas e previsões concluídas")

    def calculate_cycle_dynamics(self):
        """Calcula dinâmica de ciclo com indicadores avançados"""
        # Preparar dados para análise de ciclo
        sccs_data = []

        for empresa in self.df['EMPRESA'].unique():
            empresa_df = self.df[self.df['EMPRESA'] == empresa].copy()

            for idx, row in empresa_df.iterrows():
                sccs_data.append({
                    'EMPRESA': empresa,
                    'DATA': row['DATA'],
                    'SCCS': row.get('SCCS', 0),
                    'SALES_GROWTH': row.get('SALES_GROWTH', np.nan),
                    'RETURN_COM_EQY': row.get('RETURN_COM_EQY', np.nan),
                    'OPER_MARGIN': row.get('OPER_MARGIN', np.nan),
                    'OPERATING_ROIC': row.get('OPERATING_ROIC', np.nan),
                    'CF_FREE_CASH_FLOW': row.get('CF_FREE_CASH_FLOW', np.nan),
                    'NET_DEBT_EQTY': row.get('NET_DEBT_EQTY', np.nan)
                })

        df = pd.DataFrame(sccs_data)

        # Calcular direção (média móvel)
        df['DIRECAO'] = df.groupby('EMPRESA')['SCCS'].transform(
            lambda x: x.rolling(2, min_periods=1).mean()
        )

        # Calcular aceleração (taxa de mudança)
        df['ACELERACAO'] = df.groupby('EMPRESA')['DIRECAO'].diff().fillna(0)

        # Calcular momentum (EMA)
        df['MOMENTUM'] = df.groupby('EMPRESA')['SCCS'].transform(
            lambda x: x.ewm(span=3, adjust=False).mean()
        )

        # Determinar ciclo
        conditions = [
            (df['DIRECAO'] >= 0) & (df['ACELERACAO'] >= 0),
            (df['DIRECAO'] >= 0) & (df['ACELERACAO'] < 0),
            (df['DIRECAO'] < 0) & (df['ACELERACAO'] < 0),
            (df['DIRECAO'] < 0) & (df['ACELERACAO'] >= 0)
        ]
        choices = ['Expansão', 'Retração', 'Desaceleração', 'Recuperação']

        df['CICLO'] = np.select(conditions, choices, default='Indefinido')

        # Adicionar indicadores de trimestre
        df['TRIMESTRE'] = df['DATA'].dt.quarter
        df['ANO'] = df['DATA'].dt.year
        df['TRIMESTRE_STR'] = df['TRIMESTRE'].astype(str) + 'T' + df['ANO'].astype(str)

        # Calcular força do ciclo
        df['CICLO_FORCA'] = np.sqrt(df['DIRECAO']**2 + df['ACELERACAO']**2)

        self.results_df = df

    def generate_comprehensive_report(self):
        """Gera relatório profissional completo COM PREVISÕES"""
        if self.results_df.empty:
            self.logger.logger.error("Nenhum resultado para gerar relatório")
            return

        self.logger.logger.info("=== GERANDO RELATÓRIO PROFISSIONAL ===")

        # 1. Resumo executivo
        print("\n" + "="*100)
        print(" RELATÓRIO EXECUTIVO - ANÁLISE SCCS PROFESSIONAL v4.0")
        print("="*100)

        # 2. Métricas gerais
        print("\n📊 MÉTRICAS GERAIS")
        print("-"*50)
        print(f"• Total de empresas analisadas: {self.results_df['EMPRESA'].nunique()}")
        print(f"• Período de análise: {self.results_df['DATA'].min():%Y-%m-%d} a {self.results_df['DATA'].max():%Y-%m-%d}")
        print(f"• Total de observações: {len(self.results_df)}")

        # 3. Top performers
        last_results = self.results_df.loc[
            self.results_df.groupby('EMPRESA')['DATA'].idxmax()
        ].set_index('EMPRESA')

        top_performers = last_results.nlargest(10, 'SCCS')[['SCCS', 'CICLO', 'DIRECAO', 'ACELERACAO', 'MOMENTUM']]

        print("\n🏆 TOP 10 EMPRESAS POR SCCS")
        print("-"*50)
        print(top_performers.round(2))

        # 4. Distribuição de ciclos
        print("\n📈 DISTRIBUIÇÃO DE CICLOS ATUAL")
        print("-"*50)
        cycle_dist = last_results['CICLO'].value_counts()
        for ciclo, count in cycle_dist.items():
            pct = (count / len(last_results)) * 100
            print(f"• {ciclo}: {count} empresas ({pct:.1f}%)")

        # 5. Métricas de risco
        if self.risk_metrics:
            print("\n⚠️ MÉTRICAS DE RISCO DO PORTFOLIO")
            print("-"*50)
            for metric, value in self.risk_metrics.items():
                if isinstance(value, float):
                    print(f"• {metric.replace('_', ' ').title()}: {value:.4f}")

        # 6. Alertas e anomalias
        print("\n🚨 ALERTAS E ANOMALIAS")
        print("-"*50)

        # Empresas com mudanças bruscas
        sccs_changes = self.results_df.groupby('EMPRESA')['SCCS'].diff()
        extreme_changes = sccs_changes[abs(sccs_changes) > sccs_changes.std() * 2]

        if not extreme_changes.empty:
            print(f"• {len(extreme_changes)} mudanças extremas detectadas em SCCS")
            affected_companies = self.results_df.loc[extreme_changes.index, 'EMPRESA'].unique()
            print(f"  Empresas afetadas: {', '.join(affected_companies[:5])}")

        # 7. Histórico de ciclos
        ciclo_pivot = self.results_df.pivot_table(
            index='EMPRESA',
            columns='TRIMESTRE_STR',
            values='CICLO',
            aggfunc='first'
        )

        if not ciclo_pivot.empty:
            ciclo_pivot = ciclo_pivot.iloc[:, -8:]  # Últimos 8 trimestres

            print("\n📅 HISTÓRICO DE CICLOS (Últimos 8 Trimestres)")
            print("-"*50)
            print(ciclo_pivot.head(10))

        # 8. NOVA SEÇÃO: PREVISÕES DE PRÓXIMO CICLO
        if hasattr(self, 'predictions') and self.predictions:
            print("\n🔮 PREVISÕES DE PRÓXIMO CICLO")
            print("-"*50)

            # Top 10 empresas com maior confiança
            sorted_preds = sorted(
                self.predictions.items(),
                key=lambda x: x[1]['confidence'],
                reverse=True
            )[:10]

            for empresa, pred in sorted_preds:
                print(f"\n{empresa}:")
                print(f"  Ciclo Atual: {pred['current_cycle']}")
                print(f"  → Próximo Ciclo: {pred['predicted_cycle']}")
                print(f"  Confiança: {pred['confidence']*100:.1f}%")
                print(f"  SCCS: {pred['indicators']['SCCS']:.2f}")

            # Resumo das previsões
            print("\n📊 RESUMO DAS PREVISÕES")
            print("-"*30)
            pred_counts = {}
            for _, pred in self.predictions.items():
                cycle = pred['predicted_cycle']
                pred_counts[cycle] = pred_counts.get(cycle, 0) + 1

            for cycle, count in sorted(pred_counts.items(), key=lambda x: x[1], reverse=True):
                pct = (count / len(self.predictions)) * 100
                print(f"  {cycle}: {count} empresas ({pct:.1f}%)")

        # 9. Exportar arquivos
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Resumo executivo
        summary_filename = f'sccs_executive_summary_{timestamp}.csv'
        top_performers.to_csv(summary_filename)

        # Adicionar previsões ao DataFrame de resumo se existirem
        if self.predictions:
            predictions_df = pd.DataFrame.from_dict(self.predictions, orient='index')
            predictions_filename = f'sccs_predictions_{timestamp}.csv'
            predictions_df.to_csv(predictions_filename)
            print(f"   • Previsões: {predictions_filename}")

        # Dados completos
        if self.config.OUTPUT_FORMAT == 'excel':
            with pd.ExcelWriter(f'sccs_analysis_{timestamp}.xlsx', engine='openpyxl') as writer:
                self.results_df.to_excel(writer, sheet_name='Análise Completa', index=False)
                top_performers.to_excel(writer, sheet_name='Top Performers')
                ciclo_pivot.to_excel(writer, sheet_name='Histórico Ciclos')

                # Adicionar previsões
                if self.predictions:
                    predictions_df.to_excel(writer, sheet_name='Previsões')

                # Adicionar métricas de risco
                if self.risk_metrics:
                    pd.DataFrame([self.risk_metrics]).to_excel(
                        writer, sheet_name='Métricas de Risco', index=False
                    )
        elif self.config.OUTPUT_FORMAT == 'parquet':
            self.results_df.to_parquet(f'sccs_analysis_{timestamp}.parquet')
        else:  # CSV padrão
            self.results_df.to_csv(f'sccs_analysis_{timestamp}.csv', index=False)

        # Exportar log de auditoria
        self.logger.export_audit_trail(f'audit_trail_{timestamp}.json')

        print(f"\n✅ Relatórios exportados com sucesso!")
        print(f"   • Resumo: {summary_filename}")
        print(f"   • Análise completa: sccs_analysis_{timestamp}.{self.config.OUTPUT_FORMAT}")
        print(f"   • Auditoria: audit_trail_{timestamp}.json")

    def run_complete_pipeline(self):
        """Executa pipeline completo com todas as melhorias"""
        try:
            print("\n" + "="*100)
            print(" SCCS ANALYSIS PIPELINE v4.0 - PROFESSIONAL EDITION")
            print("="*100)

            # 1. Instalar pacotes
            print("\n📦 Etapa 1: Verificando dependências...")
            PackageManager.install_packages(['core', 'financial'])

            # 2. Carregar dados
            print("\n📊 Etapa 2: Carregando e validando dados...")
            if not self.load_and_prepare_data():
                raise Exception("Falha ao carregar dados")

            # 3. Executar análise
            print("\n🔬 Etapa 3: Executando análise SCCS, métricas e previsões...")
            self.run_analysis()

            # 4. Gerar visualizações
            print("\n📈 Etapa 4: Gerando visualizações...")

            # Dashboard interativo
            self.dashboard = self.visualizer.create_interactive_dashboard(
                self.results_df,
                self.risk_metrics
            )

            if self.dashboard:
                # Salvar dashboard
                try:
                    import plotly.io as pio
                    pio.write_html(
                        self.dashboard,
                        f'sccs_dashboard_{datetime.now():%Y%m%d_%H%M%S}.html'
                    )
                    print("   ✅ Dashboard interativo salvo")
                except:
                    print("   ⚠️ Não foi possível salvar o dashboard")

            # Gráficos de ciclo para top empresas
            top_empresas = self.results_df.groupby('EMPRESA')['SCCS'].mean().nlargest(5).index

            for empresa in top_empresas:
                print(f"\n   Gerando gráfico de ciclo para {empresa}...")
                self.visualizer.create_cycle_plot_enhanced(empresa, self.results_df)

            # 5. Gerar relatório
            print("\n📝 Etapa 5: Gerando relatório profissional...")
            self.generate_comprehensive_report()

            print("\n" + "="*100)
            print(" ✅ ANÁLISE CONCLUÍDA COM SUCESSO!")
            print(" 🔮 SISTEMA DE PREVISÃO DE CICLOS ATIVADO")
            print("="*100)

            return self.results_df, self.risk_metrics, self.dashboard, self.predictions

        except Exception as e:
            self.logger.log_error('run_complete_pipeline', e)
            print(f"\n❌ ERRO CRÍTICO: {str(e)}")
            raise

# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================
def main():
    """Função principal para execução do pipeline"""

    # Configurar warnings
    warnings.filterwarnings('ignore')

    # URL da planilha
    sheet_url = "https://docs.google.com/spreadsheets/d/1SWNjeO7DcYnpYIEbT32Jh2R5un_5b5zP3uoXEH2PvbU/edit?usp=sharing"

    # Criar configuração customizada (opcional)
    config = Config()
    config.LOG_LEVEL = 'INFO'
    config.OUTPUT_FORMAT = 'csv'  # ou 'excel', 'parquet'

    # Criar e executar pipeline
    pipeline = SCCSAnalysisPipeline(sheet_url, config)

    try:
        results_df, risk_metrics, dashboard, predictions = pipeline.run_complete_pipeline()

        # Retornar resultados para uso posterior
        return {
            'results': results_df,
            'risk_metrics': risk_metrics,
            'dashboard': dashboard,
            'predictions': predictions,
            'pipeline': pipeline
        }

    except Exception as e:
        print(f"\n❌ Erro na execução: {e}")
        return None

# =============================================================================
# EXECUÇÃO
# =============================================================================
if __name__ == "__main__":
    results = main()

    if results:
        print("\n📊 Pipeline disponível em 'results' para análises adicionais")
        print("   • results['results']: DataFrame com análise completa")
        print("   • results['risk_metrics']: Métricas de risco calculadas")
        print("   • results['dashboard']: Dashboard interativo (se disponível)")
        print("   • results['predictions']: Previsões de próximo ciclo para cada empresa")
        print("   • results['pipeline']: Objeto pipeline para análises customizadas")

