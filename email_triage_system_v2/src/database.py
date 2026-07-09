"""
Database Module for Email Triage System
Stores audit logs and analytics data using SQLite.
"""
import sqlite3
from datetime import datetime
import json
import os

class TriageDatabase:
    """Manages database operations for email triage logging."""
    
    def __init__(self, db_path=None):
        """
        Initialize database connection.
        
        Args:
            db_path (str): Path to SQLite database file
        """
        if db_path is None:
            src_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(src_dir)
            db_path = os.path.join(project_root, 'data', 'email_triage.db')
        self.db_path = db_path
        self._ensure_directory()
        self.connection = None
        self.create_tables()
    
    def _ensure_directory(self):
        """Ensure database directory exists."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
    
    def connect(self):
        """Connect to database."""
        # Enable multi-threaded access for Streamlit
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        self.connection = conn
        return conn
    
    def disconnect(self):
        """Disconnect from database."""
        if self.connection:
            self.connection.close()
    
    def create_tables(self):
        """Create database tables if they don't exist."""
        conn = self.connect()
        cursor = conn.cursor()
        
        # Email processing log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS email_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email_id TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                sender TEXT,
                subject TEXT,
                category TEXT,
                priority TEXT,
                sentiment_label TEXT,
                sentiment_score REAL,
                urgency_level TEXT,
                department TEXT,
                processing_time_ms INTEGER,
                success BOOLEAN,
                error_message TEXT
            )
        ''')
        
        # Analytics summary table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analytics_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE,
                category TEXT,
                total_count INTEGER,
                high_priority_count INTEGER,
                avg_sentiment_score REAL,
                avg_processing_time_ms INTEGER
            )
        ''')
        
        conn.commit()
        self.disconnect()
    
    def log_email_processing(self, email_data, classification_result, processing_time_ms, success=True, error=None):
        """
        Log email processing to database.
        
        Args:
            email_data (dict): Email data from monitor
            classification_result (dict): ML classification and sentiment results
            processing_time_ms (int): Processing time in milliseconds
            success (bool): Whether processing succeeded
            error (str): Error message if failed
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO email_log (
                email_id, sender, subject, category, priority,
                sentiment_label, sentiment_score, urgency_level,
                department, processing_time_ms, success, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            email_data.get('id'),
            email_data.get('from'),
            email_data.get('subject'),
            classification_result.get('category'),
            classification_result.get('priority'),
            classification_result.get('sentiment', {}).get('label'),
            classification_result.get('sentiment', {}).get('compound'),
            classification_result.get('urgency', {}).get('level'),
            classification_result.get('department'),
            processing_time_ms,
            success,
            error
        ))
        
        conn.commit()
        self.disconnect()
    
    def get_recent_emails(self, limit=50):
        """Get recent processed emails."""
        conn = self.connect()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM email_log
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        self.disconnect()
        
        return [dict(row) for row in rows]
    
    def get_category_distribution(self, days=7):
        """Get category distribution for last N days."""
        conn = self.connect()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT category, COUNT(*) as count
            FROM email_log
            WHERE timestamp >= datetime('now', '-' || ? || ' days')
            GROUP BY category
            ORDER BY count DESC
        ''', (days,))
        
        rows = cursor.fetchall()
        self.disconnect()
        
        return {row['category']: row['count'] for row in rows}
    
    def get_priority_distribution(self, days=7):
        """Get priority distribution for last N days."""
        conn = self.connect()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT priority, COUNT(*) as count
            FROM email_log
            WHERE timestamp >= datetime('now', '-' || ? || ' days')
            GROUP BY priority
            ORDER BY 
                CASE priority
                    WHEN 'high' THEN 1
                    WHEN 'medium' THEN 2
                    WHEN 'low' THEN 3
                END
        ''', (days,))
        
        rows = cursor.fetchall()
        self.disconnect()
        
        return {row['priority']: row['count'] for row in rows}
    
    def get_sentiment_stats(self, days=7):
        """Get sentiment statistics for last N days."""
        conn = self.connect()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                sentiment_label,
                COUNT(*) as count,
                AVG(sentiment_score) as avg_score
            FROM email_log
            WHERE timestamp >= datetime('now', '-' || ? || ' days')
            AND sentiment_label IS NOT NULL
            GROUP BY sentiment_label
        ''', (days,))
        
        rows = cursor.fetchall()
        self.disconnect()
        
        return [dict(row) for row in rows]
    
    def get_processing_stats(self, days=7):
        """Get processing statistics."""
        conn = self.connect()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_processed,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                AVG(processing_time_ms) as avg_processing_time,
                MAX(processing_time_ms) as max_processing_time
            FROM email_log
            WHERE timestamp >= datetime('now', '-' || ? || ' days')
        ''', (days,))
        
        row = cursor.fetchone()
        self.disconnect()
        
        return dict(row) if row else {}
    
    def get_hourly_volume(self, days=1):
        """Get email volume by hour."""
        conn = self.connect()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                strftime('%H', timestamp) as hour,
                COUNT(*) as count
            FROM email_log
            WHERE timestamp >= datetime('now', '-' || ? || ' days')
            GROUP BY hour
            ORDER BY hour
        ''', (days,))
        
        rows = cursor.fetchall()
        self.disconnect()
        
        return {row['hour']: row['count'] for row in rows}


def test_database():
    """Test database operations."""
    
    print("="*70)
    print("DATABASE TEST")
    print("="*70)
    
    db = TriageDatabase('data/test_email_triage.db')
    
    # Test logging
    test_email = {
        'id': 'test123',
        'from': 'test@example.com',
        'subject': 'Test Email'
    }
    
    test_classification = {
        'category': 'support',
        'priority': 'high',
        'sentiment': {'label': 'negative', 'compound': -0.5},
        'urgency': {'level': 'high'},
        'department': 'support@company.com'
    }
    
    db.log_email_processing(test_email, test_classification, 150, success=True)
    print("[OK] Logged test email")
    
    # Get recent emails
    recent = db.get_recent_emails(limit=5)
    print(f"\n[OK] Retrieved {len(recent)} recent emails")
    
    # Get stats
    stats = db.get_processing_stats(days=30)
    print(f"\n[OK] Processing stats: {stats}")
    
    print("\n" + "="*70)
    print("Database test complete!")


if __name__ == "__main__":
    test_database()
