"""
Dynamic visualization module for creating interactive charts based on extracted data.
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional
import pandas as pd
import json
import re

class DataVisualizer:
    """Handles creation of interactive visualizations for numerical data."""
    
    @staticmethod
    def group_related_metrics(data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Groups related metrics based on category, unit, and time period.
        
        Args:
            data: List of dictionaries containing metric data
            
        Returns:
            Dictionary of grouped metrics
        """
        groups = {}
        
        # First pass: identify potential groups based on units and categories
        for item in data:
            key = f"{item.get('category', '')}_{item.get('unit', '')}"
            if key not in groups:
                groups[key] = []
            groups[key].append(item)
        
        # Second pass: validate groups for compatibility
        valid_groups = {}
        for key, items in groups.items():
            # Check if items have compatible data types and units
            if len(items) > 1:
                first_item = items[0]
                compatible = all(
                    isinstance(item.get('value'), type(first_item.get('value'))) and
                    item.get('unit') == first_item.get('unit')
                    for item in items
                )
                if compatible:
                    valid_groups[key] = items
        
        return valid_groups

    @staticmethod
    def suggest_visualization(data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Suggests appropriate visualization types for the given data.
        
        Args:
            data: List of dictionaries containing metric data
            
        Returns:
            Dictionary containing visualization suggestions
        """
        # Group related metrics
        grouped_data = DataVisualizer.group_related_metrics(data)
        
        if not grouped_data:
            return {
                "chart_type": "bar",
                "title": "Data Visualization",
                "x_axis": "Categories",
                "y_axis": "Values",
                "explanation": "Default visualization for numerical data",
                "data_group": None
            }
        
        # Find the group with the most compatible items
        best_group_key = max(grouped_data.keys(), key=lambda k: len(grouped_data[k]))
        best_group = grouped_data[best_group_key]
        
        # Determine chart type based on data characteristics
        has_time = any('time' in item and item['time'] for item in best_group)
        is_percentage = best_group[0].get('unit') == '%'
        is_comparison = len(best_group) > 1
        
        if has_time:
            chart_type = "line"
            x_label = "Time Period"
        elif is_comparison:
            chart_type = "bar"
            x_label = "Metrics"
        else:
            chart_type = "bar"
            x_label = "Category"
        
        # Get unit for y-axis label
        unit = best_group[0].get('unit', '')
        y_label = f"Value ({unit})" if unit else "Value"
        
        # Generate appropriate title
        category = best_group[0].get('category', '')
        title = f"{category} Metrics" if category else "Data Visualization"
        
        return {
            "chart_type": chart_type,
            "title": title,
            "x_axis": x_label,
            "y_axis": y_label,
            "explanation": f"Showing {len(best_group)} related metrics from category '{category}'",
            "data_group": best_group
        }

    @staticmethod
    def create_chart(data: List[Dict[str, Any]], chart_type: str,
                    title: str = "", x_axis: str = "", y_axis: str = "") -> go.Figure:
        """
        Creates a Plotly chart based on the specified type and data.
        
        Args:
            data: List of dictionaries containing metric data
            chart_type: Type of chart to create
            title: Chart title
            x_axis: X-axis label
            y_axis: Y-axis label
            
        Returns:
            Plotly figure object
        """
        # Convert data to proper format for visualization
        df = pd.DataFrame(data)
        
        # Clean and standardize values
        if 'value' in df.columns:
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        if chart_type == "bar":
            fig = px.bar(
                df,
                x='label',
                y='value',
                title=title,
                labels={'label': x_axis, 'value': y_axis},
                text='value'
            )
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            
        elif chart_type == "line":
            fig = px.line(
                df,
                x='time',
                y='value',
                title=title,
                labels={'time': x_axis, 'value': y_axis},
                markers=True
            )
            
        elif chart_type == "pie":
            fig = px.pie(
                df,
                values='value',
                names='label',
                title=title
            )
            
        elif chart_type == "scatter":
            fig = px.scatter(
                df,
                x='label',
                y='value',
                title=title,
                labels={'label': x_axis, 'value': y_axis}
            )
            
        else:
            fig = px.bar(
                df,
                x='label',
                y='value',
                title=title,
                labels={'label': x_axis, 'value': y_axis}
            )
        
        fig.update_layout(
            template="plotly_white",
            showlegend=True,
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center'
            }
        )
        
        return fig

    @staticmethod
    def extract_numerical_data(text: str) -> List[Dict[str, Any]]:
        """
        Extracts numerical data from text using LLM.
        
        Args:
            text: Text containing numerical information
            
        Returns:
            List of dictionaries containing extracted data
        """
        from src.llm.query_engine import QueryEngine
        
        prompt = f"""
        Extract numerical data from the following text and format it as a JSON array.
        Group related metrics together and ensure they have compatible units and formats.
        Each object should contain:
        - label: Description of the data point
        - value: Numerical value (without unit)
        - unit: Unit of measurement (e.g., "%", "USD", etc.)
        - category: Group or category the data belongs to
        - time: Time period (if applicable)
        
        Text:
        {text}
        
        Return only the JSON array, no additional text.
        Example:
        [
            {{
                "label": "Revenue Q1",
                "value": 1000000,
                "unit": "USD",
                "category": "Financial",
                "time": "2025-Q1"
            }}
        ]
        
        Make sure all items in the same category have compatible units and data types.
        """
        
        engine = QueryEngine()
        extraction = engine.generate_response(query=prompt, context=[""], intent="DATA_EXTRACTION")
        
        try:
            data = json.loads(extraction)
            # Group the data by category and unit for visualization
            grouped_data = DataVisualizer.group_related_metrics(data)
            # Return the largest compatible group
            if grouped_data:
                best_group_key = max(grouped_data.keys(), key=lambda k: len(grouped_data[k]))
                return grouped_data[best_group_key]
            return []
        except:
            return []

    @staticmethod
    def convert_to_dataframe(data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Converts extracted data into a pandas DataFrame.
        
        Args:
            data: List of dictionaries containing extracted data
            
        Returns:
            Pandas DataFrame
        """
        return pd.DataFrame(data)

    @staticmethod
    def create_time_spent_chart(data: Dict[str, float]) -> go.Figure:
        """
        Create a bar chart for time spent gains across platforms.
        
        Args:
            data: Dictionary mapping platform names to percentage gains
            
        Returns:
            Plotly figure object
        """
        platforms = list(data.keys())
        gains = list(data.values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=platforms,
                y=gains,
                text=[f"{g}%" for g in gains],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Time Spent Gains by Platform",
            xaxis_title="Platform",
            yaxis_title="Gain (%)",
            yaxis_range=[0, max(gains) * 1.2],  # Add 20% padding
            showlegend=False,
            template="plotly_white"
        )
        
        return fig

    @staticmethod
    def create_revenue_forecast_chart(data: Dict[str, float]) -> go.Figure:
        """
        Create a line chart for revenue forecasts.
        
        Args:
            data: Dictionary mapping time periods to revenue values
            
        Returns:
            Plotly figure object
        """
        periods = list(data.keys())
        revenues = list(data.values())
        
        fig = go.Figure(data=[
            go.Scatter(
                x=periods,
                y=revenues,
                mode='lines+markers+text',
                text=[f"${r}B" for r in revenues],
                textposition='top center',
            )
        ])
        
        fig.update_layout(
            title="Revenue Forecast",
            xaxis_title="Period",
            yaxis_title="Revenue (Billions USD)",
            template="plotly_white"
        )
        
        return fig

    @staticmethod
    def create_headcount_chart(data: Dict[str, int]) -> go.Figure:
        """
        Create a bar chart for headcount changes.
        
        Args:
            data: Dictionary mapping departments/categories to headcount numbers
            
        Returns:
            Plotly figure object
        """
        categories = list(data.keys())
        counts = list(data.values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=counts,
                text=counts,
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Headcount by Category",
            xaxis_title="Category",
            yaxis_title="Number of Employees",
            showlegend=False,
            template="plotly_white"
        )
        
        return fig

    @staticmethod
    def create_attribution_chart(value: float) -> go.Figure:
        """
        Create a gauge chart for attribution metrics.
        
        Args:
            value: Attribution percentage value
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Attribution Rate"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 33], 'color': "lightgray"},
                    {'range': [33, 66], 'color': "gray"},
                    {'range': [66, 100], 'color': "darkgray"}
                ],
            }
        ))
        
        fig.update_layout(
            title="Attribution Metrics",
            template="plotly_white"
        )
        
        return fig

    @staticmethod
    def create_regulatory_landscape_chart(data: Dict[str, Any]) -> go.Figure:
        """
        Create a timeline or status chart for regulatory landscape.
        
        Args:
            data: Dictionary containing regulatory information
            
        Returns:
            Plotly figure object
        """
        # Convert regulatory data into a format suitable for visualization
        df = pd.DataFrame(data)
        
        fig = px.timeline(df, x_start="start_date", x_end="end_date", y="regulation",
                         color="status", title="Regulatory Landscape Timeline")
        
        fig.update_layout(
            xaxis_title="Timeline",
            yaxis_title="Regulation",
            template="plotly_white"
        )
        
        return fig

    @staticmethod
    def display_chart(fig: go.Figure, title: Optional[str] = None, height: int = 400):
        """
        Display a Plotly chart in Streamlit with optional configurations.
        
        Args:
            fig: Plotly figure object to display
            title: Optional title to display above the chart
            height: Height of the chart in pixels
        """
        if title:
            st.subheader(title)
        
        st.plotly_chart(fig, use_container_width=True, height=height)

    @staticmethod
    def parse_metrics_data(raw_data: str) -> Dict[str, Any]:
        """
        Parse raw metrics data into structured format for visualization.
        
        Args:
            raw_data: Raw string containing metrics data
            
        Returns:
            Dictionary containing structured data for different chart types
        """
        # Initialize structured data
        structured_data = {
            'time_spent': {},
            'revenue_forecast': {},
            'headcount': {},
            'attribution': 0.0,
            'regulatory': []
        }
        
        # Parse raw data line by line
        lines = raw_data.split('\n')
        for line in lines:
            line = line.strip()
            
            # Time spent gains
            if 'Facebook' in line and '%' in line:
                structured_data['time_spent']['Facebook'] = float(line.split('%')[0].split(':')[-1].strip())
            elif 'Instagram' in line and '%' in line:
                structured_data['time_spent']['Instagram'] = float(line.split('%')[0].split(':')[-1].strip())
            
            # Revenue forecast
            if 'revenue forecast' in line.lower():
                value = float(line.split('$')[1].split()[0])
                structured_data['revenue_forecast']['2025'] = value
            
            # Headcount
            if 'headcount' in line.lower() and 'increase' in line.lower():
                value = int(''.join(filter(str.isdigit, line)))
                structured_data['headcount']['Q1 Increase'] = value
            
            # Attribution
            if 'incremental' in line.lower() and '%' in line:
                value = float(line.split('%')[0].split()[-1])
                structured_data['attribution'] = value
        
        return structured_data 