import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime

class TicketPredictor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        if 'Created' in self.df.columns:
            self.df['Created'] = pd.to_datetime(self.df['Created'])
            
        # Refine Item logic for General Requests
        if 'Item' in self.df.columns:
            mask = self.df['Item'] == 'General Request'
            if bool(mask.any()):
                s1, s2 = 'Sub Category 1', 'Sub Category 2'
                if s1 in self.df.columns or s2 in self.df.columns:
                    def get_refined_item(row):
                        parts = [str(row[c]).strip() for c in [s1, s2] if c in row and pd.notnull(row[c]) and str(row[c]).strip()]
                        return " - ".join(parts) if parts else 'General Request'
                    self.df.loc[mask, 'Item'] = self.df[mask].apply(get_refined_item, axis=1)
    
    def predict_next_year(self) -> dict:
        if 'Created' not in self.df.columns:
            return {
                'predictions_by_item': {},
                'predicted_heavy_hitters': [],
                'prediction_period': {'start': None, 'end': None}
            }
        
        top_items = self.df['Item'].value_counts().head(10).index.tolist()
        predictions = {}
        
        for item in top_items:
            item_pred = self._predict_item_volume(item)
            if item_pred:
                predictions[item] = item_pred
        
        predicted_totals = {
            item: sum(p['predicted_count'] for p in preds['monthly_predictions'])
            for item, preds in predictions.items()
        }
        
        ranked_predictions = sorted(
            predicted_totals.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        from datetime import timedelta
        return {
            'predictions_by_item': predictions,
            'predicted_heavy_hitters': [
                {
                    'rank': i + 1,
                    'item': item,
                    'predicted_annual_count': int(count),
                    'current_annual_rate': self._get_current_annual_rate(item)
                }
                for i, (item, count) in enumerate(ranked_predictions)
            ],
            'prediction_period': {
                'start': (datetime.now() + timedelta(days=30)).strftime('%Y-%m'),
                'end': (datetime.now() + timedelta(days=365)).strftime('%Y-%m')
            }
        }
    
    def _predict_item_volume(self, item: str) -> dict:
        item_df = self.df[self.df['Item'] == item].copy()
        
        item_df['YearMonth'] = item_df['Created'].dt.to_period('M')
        monthly = item_df.groupby('YearMonth').size().reset_index()
        monthly.columns = ['period', 'count']
        monthly['period_num'] = range(len(monthly))
        
        if len(monthly) < 3:
            return None
        
        X = monthly['period_num'].values.reshape(-1, 1)
        y = monthly['count'].values
        
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        
        model = LinearRegression()
        model.fit(X_poly, y)
        
        last_period = monthly['period_num'].max()
        future_periods = np.array(range(last_period + 1, last_period + 13)).reshape(-1, 1)
        future_poly = poly.transform(future_periods)
        future_predictions = model.predict(future_poly)
        
        future_predictions = np.maximum(future_predictions, 0)
        
        last_date = monthly['period'].max().to_timestamp()
        future_months = [
            (last_date + pd.DateOffset(months=i+1)).strftime('%Y-%m')
            for i in range(12)
        ]
        
        return {
            'historical': [
                {'period': str(row['period']), 'count': int(row['count'])}
                for _, row in monthly.iterrows()
            ],
            'monthly_predictions': [
                {'period': month, 'predicted_count': int(round(pred))}
                for month, pred in zip(future_months, future_predictions)
            ],
            'trend': 'increasing' if future_predictions[-1] > future_predictions[0] else 'decreasing',
            'confidence': self._calculate_confidence(model, X_poly, y)
        }
    
    def _get_current_annual_rate(self, item: str) -> int:
        item_df = self.df[self.df['Item'] == item]
        days_span = (self.df['Created'].max() - self.df['Created'].min()).days
        if days_span == 0:
            return len(item_df)
        return int(len(item_df) * 365 / days_span)
    
    def _calculate_confidence(self, model, X, y) -> str:
        r2 = model.score(X, y)
        if r2 > 0.8:
            return 'high'
        elif r2 > 0.5:
            return 'medium'
        return 'low'
