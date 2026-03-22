"""
PTB-XL ECG Parser: Xử lý file .HEA (header) + .DAT (data) từ PTB-XL dataset
"""
import re
import struct
import numpy as np
from pathlib import Path


class PTBXLParser:
    """Parser cho định dạng WFDB (MIT-BIH format) của PTB-XL"""
    
    @staticmethod
    def parse_header(hea_path):
        """
        Parse file .HEA để lấy metadata
        
        Args:
            hea_path: đường dẫn đến file .HEA
            
        Returns:
            dict: thông tin header gồm:
                - num_leads: số kênh (leads)
                - sampling_rate: tần số lấy mẫu (Hz)
                - num_samples: số mẫu dữ liệu
                - leads: tên các kênh (I, II, III, AVR, AVL, AVF, V1-V6)
                - gain: hệ số chuyển đổi từ raw → mV
        """
        hea_path = Path(hea_path)
        metadata = {}
        
        with open(hea_path, 'r') as f:
            lines = f.readlines()
        
        # Parse dòng đầu tiên: record_name num_leads sampling_rate num_samples
        first_line = lines[0].strip().split()
        metadata['record_name'] = first_line[0]
        metadata['num_leads'] = int(first_line[1])
        metadata['sampling_rate'] = int(first_line[2])
        metadata['num_samples'] = int(first_line[3])
        
        # Parse thông tin từng kênh
        metadata['leads'] = []
        metadata['gains'] = []
        metadata['baselines'] = []
        
        for i in range(1, metadata['num_leads'] + 1):
            line = lines[i].strip()
            parts = re.split(r'[\s\()/]+', line)
            
            # Tính gain (LSB → mV)
            # Format: "gain(baseline)/units" e.g. "1000.0(0)/mV"
            gain_match = re.search(r'([\d.]+)\((\d+)\)', line)
            if gain_match:
                gain = float(gain_match.group(1))
                baseline = int(gain_match.group(2))
                metadata['gains'].append(gain)
                metadata['baselines'].append(baseline)
            else:
                metadata['gains'].append(1.0)
                metadata['baselines'].append(0)
            
            # Lead name (I, II, III, AVR, AVL, AVF, V1-V6)
            lead_name = parts[-1] if len(parts) > 0 else f"Lead_{i}"
            metadata['leads'].append(lead_name)
        
        return metadata
    
    @staticmethod
    def read_data(dat_path, hea_metadata):
        """
        Đọc file .DAT (dữ liệu nhị phân)
        
        Args:
            dat_path: đường dẫn đến file .DAT
            hea_metadata: metadata từ parse_header()
            
        Returns:
            numpy array: shape (num_samples, num_leads) - dữ liệu ECG tính bằng mV
        """
        dat_path = Path(dat_path)
        num_samples = hea_metadata['num_samples']
        num_leads = hea_metadata['num_leads']
        gains = hea_metadata['gains']
        baselines = hea_metadata['baselines']
        
        # Đọc dữ liệu nhị phân 16-bit signed, little-endian, multiplexed
        with open(dat_path, 'rb') as f:
            raw_data = f.read()
        
        # Unpack: '<' = little-endian, 'h' = signed short (16-bit)
        num_values = len(raw_data) // 2
        raw_array = struct.unpack(f'<{num_values}h', raw_data)
        
        # Reshape: multiplexed format = [lead0_sample0, lead1_sample0, ..., lead0_sample1, ...]
        raw_array = np.array(raw_array, dtype=np.float32)
        data = raw_array.reshape((num_samples, num_leads))
        
        # Chuyển đổi từ digital → mV
        ecg_data = np.zeros_like(data)
        for lead_idx in range(num_leads):
            ecg_data[:, lead_idx] = (data[:, lead_idx] - baselines[lead_idx]) / gains[lead_idx]
        
        return ecg_data
    
    @staticmethod
    def load_record(record_dir, record_name):
        """
        Load toàn bộ record từ .HEA + .DAT
        
        Args:
            record_dir: thư mục chứa record (e.g. data/ptbxl/records100/00000/)
            record_name: tên record (e.g. "00001_lr")
            
        Returns:
            tuple: (ecg_data, metadata)
                - ecg_data: array shape (num_samples, num_leads)
                - metadata: dict chứa thông tin record
        """
        record_dir = Path(record_dir)
        hea_path = record_dir / f"{record_name}.hea"
        dat_path = record_dir / f"{record_name}.dat"
        
        if not hea_path.exists() or not dat_path.exists():
            raise FileNotFoundError(f"Record {record_name} not found in {record_dir}")
        
        metadata = PTBXLParser.parse_header(hea_path)
        ecg_data = PTBXLParser.read_data(dat_path, metadata)
        
        return ecg_data, metadata
    
    @staticmethod
    def preprocess_for_model(ecg_signal, target_length=256, lead_index=None):
        """
        Tiền xử lý tín hiệu ECG để phù hợp với mô hình
        
        Args:
            ecg_signal: array shape (num_samples,) hoặc (num_samples, num_leads)
            target_length: độ dài mục tiêu (mặc định 256)
            lead_index: nếu 2D, chọn lead nào (mặc định 0 = Lead I)
            
        Returns:
            numpy array: dữ liệu chuẩn hóa (0-1), shape (target_length,)
        """
        # Nếu 2D, trích xuất một kênh
        if len(ecg_signal.shape) == 2:
            if lead_index is None:
                lead_index = 0  # Lead I
            signal = ecg_signal[:, lead_index]
        else:
            signal = ecg_signal
        
        # Downsampling hoặc interpolation để vừa với target_length
        if len(signal) > target_length:
            # Downsampling bằng cách lấy mẫu đều
            indices = np.linspace(0, len(signal) - 1, target_length, dtype=int)
            signal = signal[indices]
        elif len(signal) < target_length:
            # Padding với zero
            signal = np.pad(signal, (0, target_length - len(signal)), mode='constant')
        
        # Chuẩn hóa min-max: [0, 1]
        signal_min = np.min(signal)
        signal_max = np.max(signal)
        
        if signal_max > signal_min:
            signal_normalized = (signal - signal_min) / (signal_max - signal_min)
        else:
            signal_normalized = signal  # Nếu toàn 0 hoặc hằng số
        
        return signal_normalized.astype(np.float32)


# Ví dụ sử dụng
if __name__ == '__main__':
    # Test load một record
    parser = PTBXLParser()
    
    record_dir = Path('data/ptbxl/records100/00000')
    record_name = '00001_lr'
    
    ecg_data, metadata = parser.load_record(record_dir, record_name)
    
    print(f"Record: {metadata['record_name']}")
    print(f"Sampling rate: {metadata['sampling_rate']} Hz")
    print(f"Num samples: {metadata['num_samples']}")
    print(f"Leads: {', '.join(metadata['leads'])}")
    print(f"ECG data shape: {ecg_data.shape}")
    
    # Tiền xử lý cho mô hình
    processed = parser.preprocess_for_model(ecg_data, target_length=256, lead_index=0)
    print(f"Processed signal shape: {processed.shape}")
    print(f"Processed signal range: [{processed.min():.4f}, {processed.max():.4f}]")
