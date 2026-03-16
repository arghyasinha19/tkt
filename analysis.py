import pandas as pd
import numpy as np

class TicketAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._preprocess()
    
    def _preprocess(self):
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
        
        self.df['Item'] = self.df['Item'].fillna('Unspecified')
        self.df['Category'] = self.df['Category'].fillna('Unspecified')
        self.df['Short Description'] = self.df['Short Description'].fillna('')
        
        if 'Created' in self.df.columns:
            self.df['Year'] = self.df['Created'].dt.year
            self.df['Month'] = self.df['Created'].dt.month
            self.df['Week'] = self.df['Created'].dt.isocalendar().week
            self.df['YearMonth'] = self.df['Created'].dt.to_period('M').astype(str)
        
        if 'Effort_Hours' not in self.df.columns:
            self._estimate_effort()
    
    def _estimate_effort(self):
        effort_map = {
            'Hardware': 2.0,
            'Software': 1.5,
            'Network': 2.5,
            'Access': 0.5,
            'Email': 0.75,
            'Password': 0.25,
            'Account': 0.5,
            'Printer': 1.0,
            'VPN': 1.5,
            'Unspecified': 1.0
        }
        self.df['Effort_Hours'] = self.df['Category'].map(
            lambda x: effort_map.get(x, 1.0)
        )
    
    def get_heavy_hitters(self, top_n: int = 10) -> dict:
        item_counts = self.df['Item'].value_counts().head(top_n)
        total_tickets = len(self.df)
        
        heavy_hitters = []
        cumulative_pct = 0
        
        for item, count in item_counts.items():
            pct = (count / total_tickets) * 100
            cumulative_pct += pct
            heavy_hitters.append({
                'item': item,
                'count': int(count),
                'percentage': round(pct, 2),
                'cumulative_percentage': round(cumulative_pct, 2)
            })
        
        return {
            'heavy_hitters': heavy_hitters,
            'total_tickets': total_tickets,
            'top_n_coverage': round(cumulative_pct, 2)
        }
    
    def get_effort_by_item(self) -> dict:
        effort_by_item = self.df.groupby('Item').agg({
            'Effort_Hours': ['sum', 'mean', 'count']
        }).round(2)
        
        effort_by_item.columns = ['total_hours', 'avg_hours', 'ticket_count']
        effort_by_item = effort_by_item.sort_values('total_hours', ascending=False)
        
        result = []
        for item, row in effort_by_item.head(15).iterrows():
            result.append({
                'item': item,
                'total_hours': float(row['total_hours']),
                'avg_hours': float(row['avg_hours']),
                'ticket_count': int(row['ticket_count'])
            })
        
        return {
            'effort_analysis': result,
            'total_effort_hours': float(self.df['Effort_Hours'].sum()),
            'avg_effort_per_ticket': float(self.df['Effort_Hours'].mean().round(2))
        }
    
    def get_volume_trends(self, granularity: str = 'monthly') -> dict:
        if 'Created' not in self.df.columns:
            return {'overall_trend': [], 'item_trends': {}, 'granularity': granularity}
        
        if granularity == 'weekly':
            self.df['Period'] = self.df['Created'].dt.to_period('W').astype(str)
        elif granularity == 'daily':
            self.df['Period'] = self.df['Created'].dt.date.astype(str)
        else:
            self.df['Period'] = self.df['YearMonth']
        
        volume = self.df.groupby('Period').size().reset_index()
        volume.columns = ['period', 'count']
        
        top_items = self.df['Item'].value_counts().head(5).index.tolist()
        item_trends = {}
        
        for item in top_items:
            item_df = self.df[self.df['Item'] == item]
            trend = item_df.groupby('Period').size().reset_index()
            trend.columns = ['period', 'count']
            item_trends[item] = trend.to_dict('records')
        
        return {
            'overall_trend': volume.to_dict('records'),
            'item_trends': item_trends,
            'granularity': granularity
        }
    
    def get_summary(self) -> dict:
        summary = {
            'total_tickets': len(self.df),
            'unique_items': int(self.df['Item'].nunique()),
            'unique_categories': int(self.df['Category'].nunique()),
            'top_category': self.df['Category'].value_counts().idxmax(),
            'top_item': self.df['Item'].value_counts().idxmax(),
        }
        
        if 'Created' in self.df.columns:
            date_range = (self.df['Created'].max() - self.df['Created'].min()).days
            summary['date_range'] = {
                'start': self.df['Created'].min().strftime('%Y-%m-%d'),
                'end': self.df['Created'].max().strftime('%Y-%m-%d')
            }
            summary['avg_daily_tickets'] = round(len(self.df) / max(date_range, 1), 1)
        else:
            summary['date_range'] = None
            summary['avg_daily_tickets'] = None
        
        return summary
