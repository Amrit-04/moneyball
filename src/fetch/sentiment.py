"""Sentiment and developer activity data fetching.

This module fetches:
- Reddit posts and sentiment using PRAW
- GitHub commit activity using PyGithub
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List
import pandas as pd
import praw
from github import Github, RateLimitExceededException

from src.config import get_config

logger = logging.getLogger(__name__)


class RedditFetcher:
    """Fetcher for Reddit sentiment data."""
    
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """Initialize Reddit fetcher.
        
        Args:
            client_id: Reddit client ID
            client_secret: Reddit client secret
            user_agent: User agent string
        """
        self.config = get_config()
        
        client_id = client_id or self.config.reddit_client_id
        client_secret = client_secret or self.config.reddit_client_secret
        user_agent = user_agent or "Moneyball/0.1.0"
        
        if not client_id or not client_secret:
            logger.warning("Reddit credentials not provided. Sentiment fetching will be disabled.")
            self.reddit = None
        else:
            try:
                self.reddit = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent=user_agent
                )
                logger.info("Reddit API initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Reddit API: {e}")
                self.reddit = None
                
    def fetch_posts(
        self,
        subreddits: List[str],
        keywords: List[str],
        start_date: datetime,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Fetch Reddit posts mentioning target assets.
        
        Args:
            subreddits: List of subreddit names
            keywords: Keywords to search for
            start_date: Start date for posts
            limit: Maximum posts per subreddit
            
        Returns:
            DataFrame with columns: timestamp, subreddit, post_id, asset_mentioned, 
                                   title, body, score, num_comments
        """
        if not self.reddit:
            logger.warning("Reddit API not available. Returning empty DataFrame.")
            return pd.DataFrame()
            
        logger.info(f"Fetching Reddit posts from {len(subreddits)} subreddits")
        
        all_posts = []
        start_timestamp = start_date.timestamp()
        
        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Search for each keyword
                for keyword in keywords:
                    try:
                        # Search recent posts
                        for post in subreddit.search(keyword, limit=limit, time_filter='month'):
                            # Filter by date
                            if post.created_utc < start_timestamp:
                                continue
                                
                            post_data = {
                                'timestamp': datetime.fromtimestamp(post.created_utc),
                                'subreddit': subreddit_name,
                                'post_id': post.id,
                                'asset_mentioned': keyword,
                                'title': post.title,
                                'body': post.selftext if hasattr(post, 'selftext') else '',
                                'score': post.score,
                                'num_comments': post.num_comments
                            }
                            
                            all_posts.append(post_data)
                            
                        logger.info(f"Fetched posts for '{keyword}' in r/{subreddit_name}")
                        
                        # Rate limiting
                        time.sleep(1)
                        
                    except Exception as e:
                        logger.error(f"Failed to search '{keyword}' in r/{subreddit_name}: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Failed to access r/{subreddit_name}: {e}")
                continue
                
        if not all_posts:
            logger.warning("No Reddit posts fetched")
            return pd.DataFrame()
            
        df = pd.DataFrame(all_posts)
        
        # Convert timestamp to UTC
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        logger.info(f"Fetched {len(df)} Reddit posts")
        return df


class GitHubFetcher:
    """Fetcher for GitHub developer activity data."""
    
    def __init__(self, token: Optional[str] = None):
        """Initialize GitHub fetcher.
        
        Args:
            token: GitHub personal access token
        """
        self.config = get_config()
        
        token = token or self.config.github_token
        
        if not token:
            logger.warning("GitHub token not provided. Developer activity fetching will be disabled.")
            self.github = None
        else:
            try:
                self.github = Github(token)
                # Test authentication
                user = self.github.get_user()
                logger.info(f"GitHub API initialized successfully (user: {user.login})")
            except Exception as e:
                logger.error(f"Failed to initialize GitHub API: {e}")
                self.github = None
                
    def fetch_commits(
        self,
        repo_urls: List[str],
        start_date: datetime
    ) -> pd.DataFrame:
        """Fetch commit activity for cryptocurrency project repositories.
        
        Args:
            repo_urls: List of repository URLs or owner/repo strings
            start_date: Start date for commits
            
        Returns:
            DataFrame with columns: timestamp, repo_name, commit_count, contributors
        """
        if not self.github:
            logger.warning("GitHub API not available. Returning empty DataFrame.")
            return pd.DataFrame()
            
        logger.info(f"Fetching commit activity for {len(repo_urls)} repositories")
        
        all_data = []
        
        for repo_url in repo_urls:
            try:
                # Extract owner/repo from URL
                if '/' in repo_url:
                    repo_name = repo_url.split('/')[-2] + '/' + repo_url.split('/')[-1]
                    repo_name = repo_name.replace('.git', '')
                else:
                    repo_name = repo_url
                    
                repo = self.github.get_repo(repo_name)
                
                # Get commits since start_date
                commits = repo.get_commits(since=start_date)
                
                # Count commits and unique contributors
                commit_count = 0
                contributors = set()
                
                for commit in commits:
                    commit_count += 1
                    if commit.author:
                        contributors.add(commit.author.login)
                        
                data = {
                    'timestamp': datetime.now(),
                    'repo_name': repo_name,
                    'commit_count': commit_count,
                    'contributors': len(contributors)
                }
                
                all_data.append(data)
                logger.info(f"Fetched {commit_count} commits from {repo_name}")
                
                # Rate limiting
                time.sleep(1)
                
            except RateLimitExceededException:
                logger.warning("GitHub API rate limit exceeded. Waiting...")
                time.sleep(60)
                continue
            except Exception as e:
                logger.error(f"Failed to fetch commits for {repo_url}: {e}")
                continue
                
        if not all_data:
            logger.warning("No GitHub commit data fetched")
            return pd.DataFrame()
            
        df = pd.DataFrame(all_data)
        
        # Convert timestamp to UTC
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        logger.info(f"Fetched commit data for {len(df)} repositories")
        return df


def fetch_reddit_sentiment(
    subreddits: List[str],
    keywords: List[str],
    start_date: datetime,
    output_dir: Optional[Path] = None
) -> pd.DataFrame:
    """Fetch Reddit posts mentioning target assets.
    
    Args:
        subreddits: List of subreddit names
        keywords: Keywords to search for
        start_date: Start date for posts
        output_dir: Directory to save Parquet file (optional)
        
    Returns:
        DataFrame with Reddit posts
    """
    fetcher = RedditFetcher()
    df = fetcher.fetch_posts(subreddits, keywords, start_date)
    
    if output_dir and not df.empty:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"reddit_{datetime.now().strftime('%Y%m%d')}.parquet"
        filepath = output_dir / filename
        
        df.to_parquet(filepath, index=False)
        logger.info(f"Saved data to {filepath}")
        
    return df


def fetch_github_commits(
    repo_urls: List[str],
    start_date: datetime,
    output_dir: Optional[Path] = None
) -> pd.DataFrame:
    """Fetch commit activity for cryptocurrency project repositories.
    
    Args:
        repo_urls: List of repository URLs or owner/repo strings
        start_date: Start date for commits
        output_dir: Directory to save Parquet file (optional)
        
    Returns:
        DataFrame with commit activity
    """
    fetcher = GitHubFetcher()
    df = fetcher.fetch_commits(repo_urls, start_date)
    
    if output_dir and not df.empty:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"github_{datetime.now().strftime('%Y%m%d')}.parquet"
        filepath = output_dir / filename
        
        df.to_parquet(filepath, index=False)
        logger.info(f"Saved data to {filepath}")
        
    return df


def main():
    """Main function for testing sentiment and developer activity fetching."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    config = get_config()
    
    # Test Reddit fetching
    subreddits = config.get('assets.subreddits', ['cryptocurrency', 'CryptoMarkets'])
    keywords = ['bitcoin', 'ethereum', 'BTC', 'ETH']
    start_date = datetime.now() - timedelta(days=7)
    
    reddit_output_dir = config.raw_data_dir / "reddit"
    
    reddit_df = fetch_reddit_sentiment(
        subreddits=subreddits[:2],  # Test with 2 subreddits
        keywords=keywords[:2],  # Test with 2 keywords
        start_date=start_date,
        output_dir=reddit_output_dir
    )
    
    if not reddit_df.empty:
        logger.info(f"Reddit: Fetched {len(reddit_df)} posts")
        logger.info(f"Subreddits: {reddit_df['subreddit'].unique()}")
        
    # Test GitHub fetching
    crypto_repos = config.get('assets.crypto_repos', {})
    repo_urls = list(crypto_repos.values())[:3]  # Test with 3 repos
    
    github_output_dir = config.raw_data_dir / "github"
    
    github_df = fetch_github_commits(
        repo_urls=repo_urls,
        start_date=start_date,
        output_dir=github_output_dir
    )
    
    if not github_df.empty:
        logger.info(f"GitHub: Fetched data for {len(github_df)} repositories")
        logger.info(f"Total commits: {github_df['commit_count'].sum()}")


if __name__ == "__main__":
    main()
