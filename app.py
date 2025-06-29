from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import io
import base64
import json
import os

app = Flask(__name__)
CORS(app)

# Global variables to store processed data
df = None
df_scaled = None
scaler = None
kmeans = None
numerical_features = ["valence", "danceability", "energy", "tempo", "acousticness", "liveness", "speechiness", "instrumentalness"]

def load_and_process_data():
    """Load and process the music data"""
    global df, df_scaled, scaler, kmeans
    
    try:
        # Load data
        df = pd.read_csv("data.csv.zip")  # Adjust path as needed
        df = df.sample(n=5000, random_state=42).reset_index(drop=True)
        
        # Scale features
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df[numerical_features]), 
            columns=numerical_features
        )
        
        # Train clustering model
        train_data, test_data = train_test_split(df_scaled, test_size=0.2, random_state=42)
        
        # Use optimal k=5 based on elbow method
        optimal_k = 5
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        df["Cluster"] = kmeans.fit_predict(df_scaled)
        
        print(f"Data loaded successfully! {len(df)} songs processed.")
        print(f"Cluster distribution:\n{df['Cluster'].value_counts()}")
        
        return True
        
    except FileNotFoundError:
        print("Data file not found. Using sample data instead.")
        create_sample_data()
        return False
    except Exception as e:
        print(f"Error loading data: {e}")
        create_sample_data()
        return False

def create_sample_data():
    """Create sample data for demonstration"""
    global df, df_scaled, scaler, kmeans
    
    # Sample data
    sample_data = {
        'name': [
            'Camby Bolongo', 'Soul Junction', 'Bohemian Rhapsody', 'Dancing Queen', 
            'Hotel California', 'Billie Jean', 'Imagine', 'Sweet Child O\' Mine',
            'Thriller', 'Stairway to Heaven', 'Yesterday', 'Like a Rolling Stone',
            'What\'s Going On', 'Respect', 'Good Vibrations', 'Purple Haze',
            'Bridge Over Troubled Water', 'My Girl', 'Satisfaction', 'Hey Jude'
        ],
        'artists': [
            'Various Artists', 'Jimmy Smith', 'Queen', 'ABBA', 'Eagles',
            'Michael Jackson', 'John Lennon', 'Guns N\' Roses', 'Michael Jackson',
            'Led Zeppelin', 'The Beatles', 'Bob Dylan', 'Marvin Gaye',
            'Aretha Franklin', 'The Beach Boys', 'Jimi Hendrix', 'Simon & Garfunkel',
            'The Temptations', 'The Rolling Stones', 'The Beatles'
        ],
        'year': [
            1970, 1965, 1975, 1976, 1976, 1983, 1971, 1987, 1982, 1971,
            1965, 1965, 1971, 1967, 1966, 1967, 1970, 1964, 1965, 1968
        ],
        'valence': [0.8, 0.6, 0.4, 0.9, 0.3, 0.7, 0.5, 0.6, 0.8, 0.4, 0.2, 0.5, 0.6, 0.8, 0.9, 0.7, 0.3, 0.9, 0.8, 0.7],
        'danceability': [0.9, 0.7, 0.3, 0.9, 0.4, 0.8, 0.3, 0.5, 0.8, 0.3, 0.4, 0.6, 0.7, 0.8, 0.7, 0.6, 0.4, 0.8, 0.7, 0.5],
        'energy': [0.7, 0.8, 0.8, 0.8, 0.6, 0.7, 0.4, 0.9, 0.8, 0.7, 0.3, 0.7, 0.6, 0.8, 0.8, 0.9, 0.4, 0.9, 0.8, 0.6],
        'tempo': [120, 110, 72, 100, 76, 117, 75, 125, 118, 82, 97, 86, 96, 117, 127, 113, 84, 125, 118, 75],
        'acousticness': [0.1, 0.3, 0.2, 0.0, 0.4, 0.0, 0.6, 0.0, 0.0, 0.1, 0.8, 0.2, 0.1, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
        'liveness': [0.3, 0.4, 0.3, 0.1, 0.1, 0.0, 0.1, 0.3, 0.0, 0.1, 0.2, 0.2, 0.1, 0.0, 0.1, 0.3, 0.1, 0.4, 0.0, 0.1],
        'speechiness': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0],
        'instrumentalness': [0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Scale features
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df[numerical_features]), 
        columns=numerical_features
    )
    
    # Create clusters
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(df_scaled)
    
    print("Sample data created successfully!")

def recommend_songs(song_name, num_recommendations=5):
    """Recommend songs based on clustering and similarity"""
    try:
        # Find songs with partial matching
        matching_songs = df[df['name'].str.contains(song_name, case=False, na=False)]
        
        if matching_songs.empty:
            return None
        
        # Use the first match
        song_data = matching_songs.iloc[0]
        song_cluster = song_data['Cluster']
        
        # Get all songs in the same cluster
        same_cluster_songs = df[df['Cluster'] == song_cluster].copy()
        
        if len(same_cluster_songs) <= 1:
            return pd.DataFrame()
        
        # Calculate cosine similarity
        song_features = df_scaled.loc[song_data.name:song_data.name, numerical_features]
        cluster_indices = same_cluster_songs.index
        cluster_features = df_scaled.loc[cluster_indices, numerical_features]
        
        # Compute similarities
        similarities = cosine_similarity(song_features, cluster_features)[0]
        
        # Create similarity dataframe
        same_cluster_songs = same_cluster_songs.copy()
        same_cluster_songs['similarity'] = similarities
        
        # Remove the input song and sort by similarity
        recommendations = same_cluster_songs[
            same_cluster_songs['name'] != song_data['name']
        ].sort_values('similarity', ascending=False).head(num_recommendations)
        
        return recommendations[['name', 'year', 'artists', 'similarity', 'Cluster']]
        
    except Exception as e:
        print(f"Error in recommend_songs: {e}")
        return pd.DataFrame()

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Music Recommendation System</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            text-align: center;
            margin-bottom: 40px;
            color: white;
        }

        .header h1 {
            font-size: 3rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }

        .header p {
            font-size: 1.2rem;
            opacity: 0.9;
        }

        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 40px;
        }

        .card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }

        .search-section h2 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.8rem;
        }

        .search-container {
            position: relative;
            margin-bottom: 20px;
        }

        .search-input {
            width: 100%;
            padding: 15px 50px 15px 20px;
            border: 2px solid #e1e5e9;
            border-radius: 50px;
            font-size: 16px;
            transition: all 0.3s ease;
            background: white;
        }

        .search-input:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 20px rgba(102, 126, 234, 0.2);
        }

        .search-btn {
            position: absolute;
            right: 5px;
            top: 50%;
            transform: translateY(-50%);
            background: linear-gradient(135deg, #667eea, #764ba2);
            border: none;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            color: white;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .search-btn:hover {
            transform: translateY(-50%) scale(1.1);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }

        .suggestions {
            max-height: 200px;
            overflow-y: auto;
            border: 1px solid #e1e5e9;
            border-radius: 10px;
            background: white;
            display: none;
            position: absolute;
            width: 100%;
            z-index: 1000;
            top: 100%;
            margin-top: 5px;
        }

        .suggestion-item {
            padding: 12px 20px;
            cursor: pointer;
            border-bottom: 1px solid #f0f0f0;
            transition: background 0.2s ease;
        }

        .suggestion-item:hover {
            background: #f8f9ff;
        }

        .suggestion-item:last-child {
            border-bottom: none;
        }

        .recommendations {
            margin-top: 20px;
        }

        .song-card {
            background: linear-gradient(135deg, #f8f9ff, #e8f0ff);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 15px;
            border-left: 5px solid #667eea;
            transition: all 0.3s ease;
            transform: translateY(0);
        }

        .song-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.15);
        }

        .song-title {
            font-weight: 600;
            color: #667eea;
            font-size: 1.1rem;
            margin-bottom: 5px;
        }

        .song-details {
            color: #666;
            font-size: 0.9rem;
        }

        .cluster-info {
            background: linear-gradient(135deg, #ff9a9e, #fecfef);
            color: white;
            padding: 10px 15px;
            border-radius: 20px;
            display: inline-block;
            font-size: 0.8rem;
            font-weight: 600;
            margin-top: 10px;
        }

        .analytics-section h2 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.8rem;
        }

        .chart-container {
            position: relative;
            height: 300px;
            margin-bottom: 20px;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 20px;
        }

        .stat-card {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
        }

        .stat-number {
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 5px;
        }

        .stat-label {
            font-size: 0.9rem;
            opacity: 0.9;
        }

        .loading {
            text-align: center;
            padding: 20px;
            color: #667eea;
        }

        .error {
            background: #ffe6e6;
            color: #d63031;
            padding: 15px;
            border-radius: 10px;
            border-left: 5px solid #d63031;
            margin: 10px 0;
        }

        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .stats-grid {
                grid-template-columns: 1fr;
            }
        }

        .pulse {
            animation: pulse 1.5s infinite;
        }

        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎵 Music Recommender</h1>
            <p>Discover similar songs using AI-powered clustering</p>
        </div>

        <div class="main-content">
            <div class="card search-section">
                <h2>Find Similar Songs</h2>
                <div class="search-container">
                    <input type="text" class="search-input" id="songSearch" placeholder="Search for a song..." autocomplete="off">
                    <button class="search-btn" onclick="getRecommendations()">🔍</button>
                    <div class="suggestions" id="suggestions"></div>
                </div>
                
                <div id="recommendations" class="recommendations"></div>
            </div>

            <div class="card analytics-section">
                <h2>Cluster Analytics</h2>
                <div class="chart-container">
                    <canvas id="clusterChart"></canvas>
                </div>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number" id="totalSongs">{{ total_songs }}</div>
                        <div class="stat-label">Total Songs</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number" id="totalClusters">{{ total_clusters }}</div>
                        <div class="stat-label">Clusters</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let chart;
        const clusterData = {{ cluster_data|safe }};

        // Initialize the application
        document.addEventListener('DOMContentLoaded', function() {
            initializeChart();
            setupSearchInput();
        });

        // Setup search input with auto-suggestions
        function setupSearchInput() {
            const searchInput = document.getElementById('songSearch');
            const suggestions = document.getElementById('suggestions');

            searchInput.addEventListener('input', function() {
                const query = this.value;
                if (query.length < 2) {
                    suggestions.style.display = 'none';
                    return;
                }

                fetch(`/api/search?q=${encodeURIComponent(query)}`)
                    .then(response => response.json())
                    .then(data => {
                        if (data.length > 0) {
                            suggestions.innerHTML = data.slice(0, 5).map(song => 
                                `<div class="suggestion-item" onclick="selectSong('${song.name}')">${song.name} - ${song.artists}</div>`
                            ).join('');
                            suggestions.style.display = 'block';
                        } else {
                            suggestions.style.display = 'none';
                        }
                    })
                    .catch(error => {
                        console.error('Search error:', error);
                        suggestions.style.display = 'none';
                    });
            });

            // Hide suggestions when clicking outside
            document.addEventListener('click', function(e) {
                if (!searchInput.contains(e.target) && !suggestions.contains(e.target)) {
                    suggestions.style.display = 'none';
                }
            });

            // Handle Enter key
            searchInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    getRecommendations();
                }
            });
        }

        function selectSong(songName) {
            document.getElementById('songSearch').value = songName;
            document.getElementById('suggestions').style.display = 'none';
            getRecommendations();
        }

        function getRecommendations() {
            const songName = document.getElementById('songSearch').value.trim();
            const recommendationsDiv = document.getElementById('recommendations');
            
            if (!songName) {
                recommendationsDiv.innerHTML = '<div class="error">Please enter a song name</div>';
                return;
            }

            // Show loading
            recommendationsDiv.innerHTML = '<div class="loading pulse">🎵 Finding similar songs...</div>';

            fetch(`/api/recommend?song=${encodeURIComponent(songName)}`)
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        recommendationsDiv.innerHTML = `<div class="error">${data.error}</div>`;
                        return;
                    }

                    if (data.recommendations.length === 0) {
                        recommendationsDiv.innerHTML = '<div class="error">No similar songs found.</div>';
                        return;
                    }

                    recommendationsDiv.innerHTML = `
                        <h3 style="color: #667eea; margin-bottom: 15px;">Songs similar to "${songName}":</h3>
                        ${data.recommendations.map((song, index) => `
                            <div class="song-card" style="animation-delay: ${index * 0.1}s">
                                <div class="song-title">${song.name}</div>
                                <div class="song-details">
                                    <strong>Artist:</strong> ${song.artists}<br>
                                    <strong>Year:</strong> ${song.year}<br>
                                    <strong>Similarity:</strong> ${(song.similarity * 100).toFixed(1)}%
                                </div>
                                <div class="cluster-info">Cluster ${song.Cluster}</div>
                            </div>
                        `).join('')}
                    `;

                    // Update chart with current song's cluster
                    if (data.input_cluster !== undefined) {
                        updateChart(data.input_cluster);
                    }
                })
                .catch(error => {
                    console.error('Recommendation error:', error);
                    recommendationsDiv.innerHTML = '<div class="error">Error getting recommendations. Please try again.</div>';
                });
        }

        function initializeChart() {
            const ctx = document.getElementById('clusterChart').getContext('2d');
            
            chart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: clusterData.map(d => `Cluster ${d.cluster}`),
                    datasets: [{
                        data: clusterData.map(d => d.count),
                        backgroundColor: [
                            '#667eea',
                            '#764ba2',
                            '#ff9a9e',
                            '#fecfef',
                            '#a8edea'
                        ],
                        borderWidth: 2,
                        borderColor: '#fff'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                padding: 20,
                                usePointStyle: true
                            }
                        }
                    }
                }
            });
        }

        function updateChart(highlightCluster) {
            if (chart) {
                // Reset all colors to default
                chart.data.datasets[0].backgroundColor = [
                    '#667eea',
                    '#764ba2', 
                    '#ff9a9e',
                    '#fecfef',
                    '#a8edea'
                ];
                
                // Highlight the selected cluster
                if (highlightCluster !== undefined) {
                    chart.data.datasets[0].backgroundColor[highlightCluster] = '#ff6b6b';
                }
                
                chart.update();
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Main page"""
    cluster_counts = df['Cluster'].value_counts().sort_index()
    cluster_data = [{"cluster": int(cluster), "count": int(count)} for cluster, count in cluster_counts.items()]
    
    return render_template_string(HTML_TEMPLATE, 
                                total_songs=len(df),
                                total_clusters=len(cluster_counts),
                                cluster_data=json.dumps(cluster_data))

@app.route('/api/search')
def search_songs():
    """Search for songs"""
    query = request.args.get('q', '')
    if not query:
        return jsonify([])
    
    # Search in both name and artist fields
    mask = (df['name'].str.contains(query, case=False, na=False) | 
            df['artists'].str.contains(query, case=False, na=False))
    
    results = df[mask][['name', 'artists', 'year']].head(10)
    return jsonify(results.to_dict('records'))

@app.route('/api/recommend')
def get_recommendations():
    """Get song recommendations"""
    song_name = request.args.get('song', '')
    if not song_name:
        return jsonify({"error": "Song name is required"})
    
    recommendations = recommend_songs(song_name)
    
    if recommendations is None:
        return jsonify({"error": f"Song '{song_name}' not found in database"})
    
    if recommendations.empty:
        return jsonify({"error": "No similar songs found in the same cluster"})
    
    # Get the input song's cluster for chart highlighting
    matching_songs = df[df['name'].str.contains(song_name, case=False, na=False)]
    input_cluster = matching_songs.iloc[0]['Cluster'] if not matching_songs.empty else None
    
    return jsonify({
        "recommendations": recommendations.to_dict('records'),
        "input_cluster": int(input_cluster) if input_cluster is not None else None
    })

@app.route('/api/cluster-stats')
def cluster_stats():
    """Get cluster statistics"""
    cluster_counts = df['Cluster'].value_counts().sort_index()
    cluster_data = [{"cluster": int(cluster), "count": int(count)} for cluster, count in cluster_counts.items()]
    
    return jsonify({
        "total_songs": len(df),
        "total_clusters": len(cluster_counts),
        "cluster_distribution": cluster_data
    })

@app.route('/api/song-details/<int:song_id>')
def song_details(song_id):
    """Get detailed information about a specific song"""
    if song_id >= len(df):
        return jsonify({"error": "Song not found"})
    
    song = df.iloc[song_id]
    return jsonify({
        "name": song['name'],
        "artists": song['artists'],
        "year": song['year'],
        "cluster": int(song['Cluster']),
        "features": {feature: float(song[feature]) for feature in numerical_features}
    })

if __name__ == '__main__':
    print("Starting Music Recommendation System...")
    print("Loading and processing data...")
    
    load_and_process_data()
    
    print("\nStarting Flask server...")
    print("Access the application at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)