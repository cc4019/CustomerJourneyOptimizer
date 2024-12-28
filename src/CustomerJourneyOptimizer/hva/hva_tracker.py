from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime

class HVATracker:
    def __init__(self):
        self.hva_records: Dict[str, List[Dict]] = {}
        self.hva_definitions: Dict[str, Dict] = {}

    def add_hva_definition(self, hva_id: str, name: str, description: str):
        """
        Add a new HVA definition to the tracker.

        :param hva_id: Unique identifier for the HVA
        :param name: Name of the HVA
        :param description: Description of the HVA
        """
        self.hva_definitions[hva_id] = {
            'name': name,
            'description': description
        }

    def record_hva(self, customer_id: str, hva_id: str, timestamp: Optional[datetime] = None, additional_data: Optional[Dict] = None):
        """
        Record an HVA performed by a customer.

        :param customer_id: Unique identifier for the customer
        :param hva_id: Unique identifier for the HVA
        :param timestamp: Time when the HVA was performed (default is current time)
        :param additional_data: Any additional data to record with the HVA
        """
        if hva_id not in self.hva_definitions:
            raise ValueError(f"HVA with ID {hva_id} is not defined.")

        if timestamp is None:
            timestamp = datetime.now()

        if customer_id not in self.hva_records:
            self.hva_records[customer_id] = []

        self.hva_records[customer_id].append({
            'hva_id': hva_id,
            'timestamp': timestamp,
            'additional_data': additional_data or {}
        })

    def get_customer_hva_history(self, customer_id: str) -> pd.DataFrame:
        """
        Get the HVA history for a specific customer.

        :param customer_id: Unique identifier for the customer
        :return: DataFrame with the customer's HVA history
        """
        if customer_id not in self.hva_records:
            return pd.DataFrame()

        records = self.hva_records[customer_id]
        df = pd.DataFrame(records)
        df['hva_name'] = df['hva_id'].map(lambda x: self.hva_definitions[x]['name'])
        return df.sort_values('timestamp')

    def get_hva_summary(self, hva_id: str) -> Dict:
        """
        Get a summary of a specific HVA across all customers.

        :param hva_id: Unique identifier for the HVA
        :return: Dictionary containing summary statistics
        """
        if hva_id not in self.hva_definitions:
            raise ValueError(f"HVA with ID {hva_id} is not defined.")

        hva_records = [
            record for customer_records in self.hva_records.values()
            for record in customer_records if record['hva_id'] == hva_id
        ]

        return {
            'total_occurrences': len(hva_records),
            'unique_customers': len(set(customer_id for customer_id in self.hva_records if any(record['hva_id'] == hva_id for record in self.hva_records[customer_id]))),
            'first_occurrence': min(record['timestamp'] for record in hva_records) if hva_records else None,
            'last_occurrence': max(record['timestamp'] for record in hva_records) if hva_records else None
        }

    def get_customer_hva_count(self, customer_id: str) -> Dict[str, int]:
        """
        Get the count of each HVA performed by a specific customer.

        :param customer_id: Unique identifier for the customer
        :return: Dictionary with HVA IDs as keys and counts as values
        """
        if customer_id not in self.hva_records:
            return {}

        hva_counts = {}
        for record in self.hva_records[customer_id]:
            hva_id = record['hva_id']
            hva_counts[hva_id] = hva_counts.get(hva_id, 0) + 1

        return hva_counts

    def get_top_hvas(self, n: int = 5) -> pd.DataFrame:
        """
        Get the top N most frequently performed HVAs across all customers.

        :param n: Number of top HVAs to return
        :return: DataFrame with HVA IDs, names, and occurrence counts
        """
        hva_counts = {}
        for customer_records in self.hva_records.values():
            for record in customer_records:
                hva_id = record['hva_id']
                hva_counts[hva_id] = hva_counts.get(hva_id, 0) + 1

        top_hvas = sorted(hva_counts.items(), key=lambda x: x[1], reverse=True)[:n]
        
        return pd.DataFrame([
            {'hva_id': hva_id, 'hva_name': self.hva_definitions[hva_id]['name'], 'count': count}
            for hva_id, count in top_hvas
        ])

    def get_hva_timeline(self, hva_id: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Get a timeline of occurrences for a specific HVA within a date range.

        :param hva_id: Unique identifier for the HVA
        :param start_date: Start date for the timeline
        :param end_date: End date for the timeline
        :return: DataFrame with dates and occurrence counts
        """
        if hva_id not in self.hva_definitions:
            raise ValueError(f"HVA with ID {hva_id} is not defined.")

        hva_records = [
            record for customer_records in self.hva_records.values()
            for record in customer_records 
            if record['hva_id'] == hva_id and start_date <= record['timestamp'] <= end_date
        ]

        df = pd.DataFrame(hva_records)
        if df.empty:
            return pd.DataFrame(columns=['date', 'count'])

        df['date'] = df['timestamp'].dt.date
        daily_counts = df.groupby('date').size().reset_index(name='count')
        daily_counts['date'] = pd.to_datetime(daily_counts['date'])

        # Ensure all dates in the range are included
        date_range = pd.date_range(start=start_date.date(), end=end_date.date(), freq='D')
        full_range_df = pd.DataFrame({'date': date_range})
        result = full_range_df.merge(daily_counts, on='date', how='left').fillna(0)
        result['count'] = result['count'].astype(int)

        return result