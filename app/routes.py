from flask import Blueprint, render_template, request, jsonify
from models.analyzer import SentimentAnalyzer
from models.config import ModelType
from utils.exceptions import ModelLoadError


bp = Blueprint('main', __name__)
analyzer = SentimentAnalyzer(ModelType.NAIVE_BAYES)

@bp.route('/')
def index():
    model_info = analyzer.get_model_info()
    return render_template('index.html', model_info=model_info)

@bp.route('/analyze', methods=['POST'])
def analyze():
    comment = request.form.get('comment')
    if not comment:
        return jsonify({'error': 'Comment is required'}), 400
    
    try:
        sentiment = analyzer.predict(comment)
        return jsonify({
            'sentiment': sentiment,
            'model_type': analyzer.model_type.value
        })
        
    except ModelLoadError as e:
        return jsonify({'error': f'Failed to load model: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@bp.route('/change_model', methods=['POST'])
def change_model():
    model_type = request.form.get('model_type')
    if model_type is None:
        return jsonify({'error': 'Model type is required'}), 400

    try:
        new_model_type = ModelType(model_type)
        global analyzer
        analyzer = SentimentAnalyzer(new_model_type)
        return jsonify({'success': True, 'model_type': model_type})
    except ValueError:
        return jsonify({'error': 'Invalid model type'}), 400
    except ModelLoadError as e:
        return jsonify({'error': f'Failed to load model: {str(e)}'}), 500
