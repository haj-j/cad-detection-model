import paho.mqtt.client as mqtt
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import glob
import sys
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import socket
import zeroconf
from zeroconf import ServiceBrowser, Zeroconf

class MQTTBrowser:
    def __init__(self):
        self.mqtt_service = None
        self.zeroconf = Zeroconf()
        self.browser = ServiceBrowser(self.zeroconf, "_mqtt._tcp.local.", self)
    
    def remove_service(self, zeroconf, type, name):
        pass
    
    def add_service(self, zeroconf, type, name):
        info = zeroconf.get_service_info(type, name)
        if info:
            self.mqtt_service = info
            print(f"Found MQTT broker at {info.server}:{info.port}")
    
    def update_service(self, zeroconf, type, name):
        pass
    
    def get_broker_address(self):
        if self.mqtt_service:
            return socket.inet_ntoa(self.mqtt_service.addresses[0]), self.mqtt_service.port
        return None, None

class CADMonitor:
    def __init__(self):
        self.MQTT_HOSTNAME = "LESANJELELA-PC"
        self.MQTT_PORT = 1883
        self.MQTT_TOPIC = "esp32/data/complete"
        self.RESULTS_TOPIC = "server/CADstatus"
        self.MODEL_DIR = r"C:\Users\HP\Desktop\projects\CAD\model_artifacts"
        self.ECG_MODEL_DIR = r"C:\Users\HP\Desktop\projects\CAD\model_artifacts"
        self.FS = 360
        self.SEGMENT_LENGTH = 5
        self.SEGMENT_SAMPLES = 100
        
        self.sensor_data = {
            'Age': None,
            'Weight': None,
            'Height': None,
            'Sex': None,
            'BMI': None,
            'Temperature': None,
            'Heart_rate': None,
            'SPO2': None,
            'Ecg': None,
            'ecg_confidence': None,
            'patientID': None
        }
        
        self.model = None
        self.scaler = None
        self.encoder = None
        self.ecg_model = None
        
        self.feature_order = [
            'Age', 'Weight', 'Height', 'Sex', 'BMI',
            'Temperature', 'Heart_rate', 'SPO2', 'Ecg'
        ]
        
        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"cad-monitor-{time.time()}",
            protocol=mqtt.MQTTv311,
            clean_session=True
        )
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect
        self.client.on_publish = self.on_publish

    def load_models(self):
        print("\nLoading model artifacts...")
        
        try:
            ecg_model_files = glob.glob(os.path.join(self.ECG_MODEL_DIR, 'ecg_model_*.pkl'))
            if not ecg_model_files:
                raise FileNotFoundError("No ECG model files found")
            
            latest_ecg_model = max(ecg_model_files, key=os.path.getctime)
            self.ecg_model = joblib.load(latest_ecg_model)
            print(f"Successfully loaded ECG model: {os.path.basename(latest_ecg_model)}")
        except Exception as e:
            print(f"Error loading ECG model: {str(e)}")
            return False
        
        try:
            model_files = glob.glob(os.path.join(self.MODEL_DIR, 'cad_model_*.pkl'))
            if not model_files:
                raise FileNotFoundError("No CAD model files found")
            
            latest_model = max(model_files, key=os.path.getctime)
            timestamp = os.path.basename(latest_model).split('_')[-1].split('.')[0]
            
            scaler_path = os.path.join(self.MODEL_DIR, f'scaler_{timestamp}.pkl')
            encoder_path = os.path.join(self.MODEL_DIR, f'label_encoder_{timestamp}.pkl')
            
            if not os.path.exists(scaler_path) or not os.path.exists(encoder_path):
                raise FileNotFoundError("Missing required CAD model files")
            
            self.model = joblib.load(latest_model)
            self.scaler = joblib.load(scaler_path)
            self.encoder = joblib.load(encoder_path)
            
            print("Successfully loaded CAD model components:")
            print(f"- Model: {os.path.basename(latest_model)}")
            print(f"- Scaler: scaler_{timestamp}.pkl")
            print(f"- Encoder: label_encoder_{timestamp}.pkl")
            print("\nModel expects features in this order:")
            print(self.feature_order)
            
            return True
            
        except Exception as e:
            print(f"Error loading CAD models: {str(e)}")
            return False

    def extract_ecg_features(self, signal):
        features = []
        
        features.extend([
            np.mean(signal),
            np.std(signal),
            np.min(signal),
            np.max(signal),
            np.median(signal),
            ((signal[:-1] * signal[1:]) < 0).sum()
        ])
        
        features.extend(np.percentile(signal, [10, 25, 75, 90]))
        
        fft_vals = np.abs(np.fft.rfft(signal))
        features.extend([
            np.mean(fft_vals),
            np.std(fft_vals),
            fft_vals.max(),
            fft_vals.argmax()
        ])
        
        return np.array(features)

    def process_ecg_signal(self, ecg_data):
        try:
            if len(ecg_data) != self.SEGMENT_SAMPLES:
                print(f"Warning: ECG segment has {len(ecg_data)} samples, expected {self.SEGMENT_SAMPLES}")
                return None, None
            
            ecg_segment = np.array(ecg_data, dtype=np.float32)
            ecg_segment = (ecg_segment - np.mean(ecg_segment)) / np.std(ecg_segment)
            
            ecg_features = self.extract_ecg_features(ecg_segment)
            
            prediction = self.ecg_model.predict([ecg_features])[0]
            proba = self.ecg_model.predict_proba([ecg_features])[0]
            confidence = proba[1] * 100 if prediction == 1 else proba[0] * 100
            
            return prediction, confidence
            
        except Exception as e:
            print(f"Error processing ECG signal: {str(e)}")
            return None, None

    def on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            print(f"\n[{datetime.now()}] Connected to broker")
            client.subscribe(self.MQTT_TOPIC, qos=1)
            print(f"[{datetime.now()}] Subscribed to: {self.MQTT_TOPIC}")
        else:
            print(f"\n[{datetime.now()}] Connection failed (code: {reason_code})")

    def on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties):
        print(f"\n[{datetime.now()}] Disconnected (code: {reason_code})")
        print("Attempting to reconnect...")
        time.sleep(2)
        self.connect_to_broker()

    def on_publish(self, client, userdata, mid, reason_code=None, properties=None):
        print(f"\n[{datetime.now()}] Results published to app (message ID: {mid})")

    def on_message(self, client, userdata, message):
        try:
            data = json.loads(message.payload.decode())
            print(f"\n[{datetime.now()}] Received data packet")
            
            required_fields = {
                'age': ('Age', int),
                'weight': ('Weight', float),
                'height': ('Height', float),
                'gender': ('Sex', str),
                'bmi': ('BMI', float),
                'temperature': ('Temperature', float),
                'heartRate': ('Heart_rate', float),
                'spo2': ('SPO2', float),
                'patientID': ('patientID', str)
            }
            
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                print(f"Warning: Missing fields in MQTT message: {missing_fields}")
                return
                
            ecg_result = None
            ecg_confidence = None
            
            if 'ecg' in data and isinstance(data['ecg'], list) and len(data['ecg']) >= 100:
                print(f"Processing {len(data['ecg'])} ECG samples...")
                ecg_result, ecg_confidence = self.process_ecg_signal(data['ecg'][:100])
                if ecg_result is not None:
                    print(f"ECG Analysis Result: {'CAD' if ecg_result == 1 else 'Normal'} (Confidence: {ecg_confidence:.2f}%)")
            
            for mqtt_field, (internal_field, field_type) in required_fields.items():
                try:
                    if mqtt_field == 'gender':
                        self.sensor_data[internal_field] = 1 if str(data[mqtt_field]).lower() == 'male' else 0
                    else:
                        self.sensor_data[internal_field] = field_type(data[mqtt_field])
                except (ValueError, KeyError) as e:
                    print(f"Error processing {mqtt_field}: {str(e)}")
                    return
            
            self.sensor_data['Ecg'] = ecg_result if ecg_result is not None else 0
            self.sensor_data['ecg_confidence'] = ecg_confidence if ecg_confidence is not None else 0
            
            if all(self.sensor_data[field] is not None for field in self.feature_order):
                print("\nComplete dataset received")
                self.process_data()
                
        except Exception as e:
            print(f"\n[{datetime.now()}] Message error: {str(e)}")
            import traceback
            traceback.print_exc()

    def discover_broker(self):
        print("Discovering MQTT broker via mDNS...")
        browser = MQTTBrowser()
        
        start_time = time.time()
        while time.time() - start_time < 10:
            broker_address, broker_port = browser.get_broker_address()
            if broker_address:
                print(f"Found broker at {broker_address}:{broker_port}")
                return broker_address, broker_port
            time.sleep(0.5)
        
        print("Failed to discover broker via mDNS")
        return None, None

    def connect_to_broker(self):
        try:
            print(f"Attempting connection to {self.MQTT_HOSTNAME}:{self.MQTT_PORT}")
            self.client.connect(self.MQTT_HOSTNAME, self.MQTT_PORT, 60)
            self.client.loop_start()
            time.sleep(0.5)
            return True
        except Exception as e:
            print(f"Direct connection failed: {str(e)}")
            print("Attempting mDNS discovery...")
            broker_address, broker_port = self.discover_broker()
            if broker_address:
                try:
                    self.client.connect(broker_address, broker_port, 60)
                    self.client.loop_start()
                    time.sleep(0.5)
                    return True
                except Exception as e:
                    print(f"mDNS connection failed: {str(e)}")
            return False

    def process_data(self):
        try:
            features = {
                'Age': self.sensor_data['Age'],
                'Weight': self.sensor_data['Weight'],
                'Height': self.sensor_data['Height'],
                'Sex': self.sensor_data['Sex'],
                'BMI': self.sensor_data['BMI'],
                'Temperature': self.sensor_data['Temperature'],
                'Heart_rate': self.sensor_data['Heart_rate'],
                'SPO2': self.sensor_data['SPO2'],
                'Ecg': self.sensor_data['Ecg']
            }
            
            input_df = pd.DataFrame([features])[self.feature_order]
            
            X = self.scaler.transform(input_df)
            pred = self.model.predict(X)
            proba = self.model.predict_proba(X)[0]
            status = self.encoder.inverse_transform(pred)[0]
            confidence = proba[1] * 100 if status == 'sick' else proba[0] * 100

            results = {
                'prediction': {
                    'status': status,
                    'confidence': float(confidence),
                    'risk': self._determine_risk_level(status, confidence)
                },
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'patientID': self.sensor_data['patientID']
            }

            self._send_results_to_app(results)
            self.display_results(input_df, status, confidence)
            self.reset_sensor_data()
            
        except Exception as e:
            print(f"\n[{datetime.now()}] Prediction error: {str(e)}")
            import traceback
            traceback.print_exc()

    def _determine_risk_level(self, status, confidence):
        if status == 'sick':
            return 'high' if confidence > 70 else 'moderate'
        return 'low'

    def _get_model_version(self):
        try:
            model_files = glob.glob(os.path.join(self.MODEL_DIR, 'cad_model_*.pkl'))
            latest = max(model_files, key=os.path.getctime)
            return os.path.basename(latest).split('_')[-1].split('.')[0]
        except:
            return "unknown"

    def _send_results_to_app(self, results):
        try:
            payload = json.dumps(results, indent=2)
            print(f"\nSending to app:\n{payload}")
            
            result = self.client.publish(
                topic=self.RESULTS_TOPIC,
                payload=payload,
                qos=1,
                retain=False
            )
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"[{datetime.now()}] Results queued for delivery (MID: {result.mid})")
            else:
                print(f"[{datetime.now()}] Publish failed (code: {result.rc})")
        except Exception as e:
            print(f"[{datetime.now()}] Error sending results: {str(e)}")

    def display_results(self, data, status, confidence):
        print("\n" + "="*50)
        print(" COMPREHENSIVE CAD PREDICTION RESULTS ".center(50, "="))
        print(f"\nPatient ID: {self.sensor_data['patientID']}")
        
        if self.sensor_data['Ecg'] is not None:
            ecg_status = "CAD" if self.sensor_data['Ecg'] == 1 else "Normal"
            print(f"\nECG Analysis:")
            print(f"- Status: {ecg_status}")
            print(f"- Confidence: {self.sensor_data['ecg_confidence']:.2f}%")
        
        print(f"\nOverall Analysis:")
        print(f"- Status: {status}")
        print(f"- Confidence: {confidence:.2f}%")
        print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if status == 'sick':
            if confidence > 70:
                print("\n HIGH RISK: Immediate medical attention required!")
            else:
                print("\n Moderate Risk: Schedule clinical follow-up")
        else:
            print("\n Low Risk: Continue regular monitoring")

    def reset_sensor_data(self):
        self.sensor_data = {k: None for k in self.sensor_data}
        print("\nSensor data reset - ready for next reading")

    def run(self):
        print("\n" + "="*50)
        print(" ADVANCED CORONARY ARTERY DISEASE MONITOR ".center(50, "="))
        print("="*50)
        
        if not self.load_models():
            sys.exit(1)
            
        try:
            while True:
                if self.connect_to_broker():
                    print(f"[{datetime.now()}] Waiting for data... (Ctrl+C to exit)")
                    while True:
                        time.sleep(1)
                else:
                    print("Retrying broker connection in 5 seconds...")
                    time.sleep(5)
                    
        except KeyboardInterrupt:
            print("\nShutting down...")
            self.client.loop_stop()
            self.client.disconnect()
            sys.exit(0)
        except Exception as e:
            print(f"\n[{datetime.now()}] Fatal error: {str(e)}")
            sys.exit(1)

if __name__ == "__main__":
    monitor = CADMonitor()
    monitor.run()
