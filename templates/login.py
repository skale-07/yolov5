from flask import Flask, request, render_template_string, jsonify
#Make sure to pip install firebase_admin
import firebase_admin
from firebase_admin import credentials, auth

# Initialize Firebase Admin SDK
cred = credentials.Certificate("path/to/your/serviceAccountKey.json")
firebase_admin.initialize_app(cred)

app = Flask(__name__)

@app.route('/verify-token', methods=['POST'])
def verify_token():
    data = request.get_json()
    token = data['token']

    try:
        # Verify the token
        decoded_token = auth.verify_id_token(token)
        user_id = decoded_token['uid']
        return jsonify({'status': 'success', 'user_id': user_id}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 401

if __name__ == '__main__':
    app.run(debug=True)