"""
Run improved ARIMA on all 10 stocks
"""
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import the improved ARIMA function
from run_improved_arima import run_improved_arima

COMPANIES = {
    'AAPL': 'Apple Inc.',
    'GOOGL': 'Alphabet Inc.',
    'MSFT': 'Microsoft Corporation',
    'AMZN': 'Amazon.com Inc.',
    'TSLA': 'Tesla Inc.',
    'META': 'Meta Platforms Inc.',
    'NVDA': 'NVIDIA Corporation',
    'JPM': 'JPMorgan Chase & Co.',
    'V': 'Visa Inc.',
    'WMT': 'Walmart Inc.'
}

def main():
    print("\n" + "="*70)
    print("RUNNING IMPROVED ARIMA ON ALL 10 STOCKS")
    print("="*70)
    print("\nThis will take approximately 50-100 minutes")
    print("(~5-10 minutes per stock)")
    print()
    
    input("Press Enter to start or Ctrl+C to cancel...")
    
    completed = []
    failed = []
    
    for i, (ticker, name) in enumerate(COMPANIES.items(), 1):
        print(f"\n{'='*70}")
        print(f"Processing {i}/10: {ticker} - {name}")
        print(f"{'='*70}")
        
        try:
            success = run_improved_arima(
                ticker=ticker,
                auto_tune=True,
                use_rolling=True
            )
            
            if success:
                completed.append(ticker)
                print(f"✅ {ticker} completed successfully!")
            else:
                failed.append(ticker)
                print(f"❌ {ticker} failed!")
                
        except Exception as e:
            failed.append(ticker)
            print(f"❌ {ticker} error: {str(e)}")
        
        print(f"\nProgress: {len(completed)}/10 completed, {len(failed)} failed")
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"\n✅ Successfully completed: {len(completed)}/10")
    for ticker in completed:
        print(f"   - {ticker}: {COMPANIES[ticker]}")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)}/10")
        for ticker in failed:
            print(f"   - {ticker}: {COMPANIES[ticker]}")
    
    print("\n" + "="*70)
    print("ALL DONE!")
    print("="*70)
    print("\nNow run: streamlit run app.py")
    print("to view all results in the dashboard!")

if __name__ == "__main__":
    main()