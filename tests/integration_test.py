#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cryo-EM Particle Picker - Integration Test Suite
"""
import sys
import os
import time
import tempfile
import traceback
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

class Colors:
    GREEN = ''
    RED = ''
    YELLOW = ''
    BLUE = ''
    ENDC = ''
    BOLD = ''

def print_header(text):
    print(f"\n{'=' * 60}")
    print(f"{text}")
    print(f"{'=' * 60}")

def print_test(name, passed, details="", timing=None):
    status = "[PASS]" if passed else "[FAIL]"
    time_str = f" [{timing:.3f}s]" if timing is not None else ""
    print(f"  {status} {name}{time_str}")
    if details:
        print(f"    {details}")

def run_test(module_name, test_func):
    print_header(f"Testing: {module_name}")
    start_time = time.time()
    passed = 0
    failed = 0
    try:
        results = test_func()
        for name, ok, details in results:
            if ok:
                passed += 1
            else:
                failed += 1
            print_test(name, ok, details)
    except Exception as e:
        failed += 1
        print_test(module_name, False, f"Exception: {str(e)}\n{traceback.format_exc()}")
    elapsed = time.time() - start_time
    total = passed + failed
    print(f"\n  {Colors.BOLD}Total: {passed}/{total} passed in {elapsed:.3f}s{Colors.ENDC}")
    return passed, failed

def test_imports():
    results = []
    start = time.time()
    
    modules = [
        ("yaml", "pyyaml"),
        ("numpy", "numpy"),
        ("cv2", "opencv-python"),
        ("torch", "PyTorch"),
        ("fastapi", "FastAPI"),
        ("pydantic", "Pydantic"),
        ("onnx", "ONNX"),
        ("onnxruntime", "ONNX Runtime"),
        ("PIL", "Pillow"),
    ]
    
    for import_name, display_name in modules:
        try:
            __import__(import_name)
            results.append((f"Import {display_name}", True, ""))
        except ImportError as e:
            results.append((f"Import {display_name}", False, str(e)))
    
    internal_modules = [
        "src.utils.config",
        "src.utils.logging",
        "src.utils.metrics",
        "src.utils.visualization",
        "src.io.mrc_parser",
        "src.io.stream_ops",
        "src.preprocessing.enhancement",
        "src.preprocessing.normalization",
        "src.preprocessing.pipeline",
        "src.modeling.backbone",
        "src.modeling.attention",
        "src.modeling.segmentation",
        "src.training.loss",
        "src.training.dataset",
        "src.inference.export_onnx",
        "src.inference.onnx_engine",
        "src.inference.pipeline",
        "src.postprocessing.detection",
        "src.postprocessing.refinement",
        "src.api.schemas",
        "src.api.service",
        "src.api.main",
    ]
    
    for mod_path in internal_modules:
        try:
            __import__(mod_path)
            results.append((f"Import {mod_path}", True, ""))
        except Exception as e:
            results.append((f"Import {mod_path}", False, f"{str(e)}\n{traceback.format_exc()}"))
    
    return results

def test_io_module():
    results = []
    
    try:
        from src.io.mrc_parser import MRCStreamParser, MRCHeader
        from src.io.stream_ops import zero_copy_read, stream_chunks, write_mrc
        
        results.append(("IO imports", True, ""))
    except Exception as e:
        results.append(("IO imports", False, str(e)))
        return results
    
    try:
        header = MRCHeader()
        header.nx = 512
        header.ny = 512
        header.nz = 1
        header.mode = 2
        header.mx = 512
        header.my = 512
        header.mz = 1
        header.cella = (100.0, 100.0, 100.0)
        header.cellb = (90.0, 90.0, 90.0)
        header.mapc = 1
        header.mapr = 2
        header.maps = 3
        header.dmin = -1.0
        header.dmax = 1.0
        header.dmean = 0.0
        header.ispg = 1
        header.nsymbt = 0
        header.extra = b'\x00' * 40
        header.origin = (0.0, 0.0, 0.0)
        header.map = b'MAP '
        header.machst = np.array([17, 17, 0, 0], dtype=np.uint8).tobytes()
        header.rms = 1.0
        header.nlabl = 10
        header.labels = [b'Test MRC file'] + [b''] * 9
        
        results.append(("MRCHeader create", True, ""))
    except Exception as e:
        results.append(("MRCHeader create", False, str(e)))
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.mrc', delete=False) as f:
            test_file = f.name
        
        try:
            test_data = np.random.randn(512, 512).astype(np.float32)
            write_mrc(test_file, test_data)
            
            results.append(("Write MRC file", True, f"Shape: {test_data.shape}, Size: {os.path.getsize(test_file)} bytes"))
        except Exception as e:
            os.unlink(test_file)
            results.append(("Write MRC file", False, str(e)))
            return results
        
        try:
            with MRCStreamParser(test_file, zero_copy=True) as parser:
                results.append(("MRCStreamParser open (zero-copy)", True, f"Shape: {parser.header.nx}x{parser.header.ny}x{parser.header.nz}"))
                try:
                    img = parser.get_image(0)
                    results.append(("Get full image", True, f"Shape: {img.shape}, Dtype: {img.dtype}"))
                except Exception as e:
                    results.append(("Get full image", False, str(e)))
                
                try:
                    region = parser.get_region(0, 100, 100, 200, 200)
                    results.append(("Get region (ROI)", True, f"Shape: {region.shape}, Dtype: {region.dtype}"))
                except Exception as e:
                    results.append(("Get region (ROI)", False, str(e)))
        except Exception as e:
            results.append(("MRCStreamParser open", False, str(e)))
            try:
                if os.path.exists(test_file):
                    os.unlink(test_file)
            except Exception:
                pass
            return results
        
        try:
            chunks = list(stream_chunks(test_file, chunk_size=4096))
            results.append(("Stream chunks", True, f"Num chunks: {len(chunks)}"))
        except Exception as e:
            results.append(("Stream chunks", False, str(e)))
        
        try:
            data = zero_copy_read(test_file)
            results.append(("Zero-copy read", True, f"Bytes: {len(data)}"))
        except Exception as e:
            results.append(("Zero-copy read", False, str(e)))
        
        try:
            import time
            time.sleep(0.1)
            if os.path.exists(test_file):
                os.unlink(test_file)
            results.append(("Cleanup", True, ""))
        except Exception as e:
            results.append(("Cleanup", False, str(e)))
        
    except Exception as e:
        results.append(("IO test", False, f"Unexpected: {str(e)}\n{traceback.format_exc()}"))
    
    return results

def test_preprocessing_module():
    results = []
    
    try:
        from src.preprocessing.enhancement import CLAHEEnhancer, BandpassFilter
        from src.preprocessing.normalization import PercentileNormalizer, ZScoreNormalizer
        from src.preprocessing.pipeline import PreprocessingPipeline
        
        results.append(("Preprocessing imports", True, ""))
    except Exception as e:
        results.append(("Preprocessing imports", False, str(e)))
        return results
    
    test_image = np.random.randn(512, 512).astype(np.float32)
    test_image[100:150, 100:150] += 3.0
    
    try:
        enh = CLAHEEnhancer(clip_limit=3.0, tile_grid_size=(8, 8))
        result = enh.apply(test_image)
        results.append(("CLAHE Enhancement", True, f"Input range: [{test_image.min():.2f}, {test_image.max():.2f}], Output range: [{result.min():.2f}, {result.max():.2f}]"))
    except Exception as e:
        results.append(("CLAHE Enhancement", False, str(e)))
    
    try:
        bp = BandpassFilter(low_sigma=50.0, high_sigma=1.0, use_fft=True)
        result = bp.apply(test_image)
        results.append(("FFT Bandpass Filter", True, f"Shape: {result.shape}, Dtype: {result.dtype}"))
    except Exception as e:
        results.append(("FFT Bandpass Filter", False, str(e)))
    
    try:
        bp_spatial = BandpassFilter(low_sigma=50.0, high_sigma=1.0, use_fft=False)
        result = bp_spatial.apply(test_image)
        results.append(("Spatial Bandpass Filter", True, f"Shape: {result.shape}, Dtype: {result.dtype}"))
    except Exception as e:
        results.append(("Spatial Bandpass Filter", False, str(e)))
    
    try:
        norm = PercentileNormalizer(percentile_low=1, percentile_high=99)
        result = norm.apply(test_image)
        results.append(("Percentile Normalization", True, f"Output range: [{result.min():.2f}, {result.max():.2f}]"))
    except Exception as e:
        results.append(("Percentile Normalization", False, str(e)))
    
    try:
        z_norm = ZScoreNormalizer()
        result = z_norm.apply(test_image)
        results.append(("Z-Score Normalization", True, f"Mean: {result.mean():.4f}, Std: {result.std():.4f}"))
    except Exception as e:
        results.append(("Z-Score Normalization", False, str(e)))
    
    try:
        pipeline = PreprocessingPipeline()
        p_result = pipeline.process(test_image)
        results.append(("Full Pipeline", True, 
                       f"Output shape: {p_result.image.shape}, "
                       f"Processing time: {p_result.processing_time:.2f}ms"))
    except Exception as e:
        results.append(("Full Pipeline", False, f"{str(e)}\n{traceback.format_exc()}"))
    
    return results

def test_modeling_module():
    results = []
    
    try:
        import torch
        from src.modeling.backbone import ResidualBlock, LightResNet, DecoderBlock
        from src.modeling.attention import MultiHeadAttention, MaskedSelfAttention, CBAMBlock
        from src.modeling.segmentation import LightResUNet_MSA
        
        results.append(("Modeling imports", True, ""))
    except Exception as e:
        results.append(("Modeling imports", False, str(e)))
        return results
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results.append(("Device", True, f"Using {device}"))
    
    try:
        res_block = ResidualBlock(in_channels=32, out_channels=32)
        x = torch.randn(2, 32, 64, 64)
        out = res_block(x)
        results.append(("ResidualBlock", True, f"Input: {x.shape}, Output: {out.shape}"))
    except Exception as e:
        results.append(("ResidualBlock", False, str(e)))
    
    try:
        backbone = LightResNet(in_channels=1, base_channels=32, num_blocks=4)
        x = torch.randn(1, 1, 256, 256)
        features = backbone(x)
        results.append(("LightResNet Backbone", True, 
                       f"Input: {x.shape}, "
                       f"Features: {[f.shape for f in features]}"))
    except Exception as e:
        results.append(("LightResNet Backbone", False, f"{str(e)}\n{traceback.format_exc()}"))
    
    try:
        mha = MultiHeadAttention(channels=64, num_heads=4)
        x = torch.randn(2, 64, 32, 32)
        out = mha(x)
        results.append(("MultiHeadAttention", True, f"Input: {x.shape}, Output: {out.shape}"))
    except Exception as e:
        results.append(("MultiHeadAttention", False, str(e)))
    
    try:
        msa = MaskedSelfAttention(channels=64, num_heads=4, attention_channels=64)
        x = torch.randn(2, 64, 32, 32)
        out = msa(x)
        results.append(("MaskedSelfAttention", True, f"Input: {x.shape}, Output: {out.shape}"))
    except Exception as e:
        results.append(("MaskedSelfAttention", False, f"{str(e)}\n{traceback.format_exc()}"))
    
    try:
        cbam = CBAMBlock(channels=64, reduction=16)
        x = torch.randn(2, 64, 32, 32)
        out = cbam(x)
        results.append(("CBAM Attention", True, f"Input: {x.shape}, Output: {out.shape}"))
    except Exception as e:
        results.append(("CBAM Attention", False, str(e)))
    
    try:
        model = LightResUNet_MSA(
            in_channels=1,
            out_channels=2,
            base_channels=32,
            num_res_blocks=4,
            attention_heads=4,
            attention_channels=64,
            dropout_rate=0.1,
            use_deep_supervision=True
        )
        model = model.to(device)
        x = torch.randn(1, 1, 256, 256).to(device)
        
        with torch.no_grad():
            outputs = model(x)
        
        if isinstance(outputs, tuple):
            results.append(("LightResUNet_MSA (with deep supervision)", True, 
                           f"Input: {x.shape}, "
                           f"Outputs: {[o.shape for o in outputs]}"))
        else:
            results.append(("LightResUNet_MSA", True, 
                           f"Input: {x.shape}, Output: {outputs.shape}"))
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        results.append(("Model Parameters", True, 
                       f"Total: {total_params:,}, Trainable: {trainable_params:,}"))
    except Exception as e:
        results.append(("LightResUNet_MSA", False, f"{str(e)}\n{traceback.format_exc()}"))
    
    return results

def test_training_module():
    results = []
    
    try:
        import torch
        from src.training.loss import DiceLoss, FocalLoss, DiceCELoss, DeepSupervisionLoss, TverskyLoss
        from src.training.dataset import CryoEMDataset, MRCDataModule
        
        results.append(("Training imports", True, ""))
    except Exception as e:
        results.append(("Training imports", False, str(e)))
        return results
    
    try:
        pred = torch.randn(2, 2, 32, 32)
        target = torch.randint(0, 2, (2, 32, 32))
        
        losses = [
            ("DiceLoss", DiceLoss()),
            ("FocalLoss", FocalLoss()),
            ("DiceCELoss", DiceCELoss()),
            ("TverskyLoss", TverskyLoss()),
        ]
        
        for name, loss_fn in losses:
            try:
                loss = loss_fn(pred, target)
                results.append((name, True, f"Loss: {loss.item():.4f}"))
            except Exception as e:
                results.append((name, False, str(e)))
    except Exception as e:
        results.append(("Loss functions", False, str(e)))
    
    try:
        ds_loss = DeepSupervisionLoss(DiceCELoss(), deep_weights=[0.5, 0.3, 0.2])
        pred_list = [
            torch.randn(2, 2, 32, 32),
            torch.randn(2, 2, 16, 16),
            torch.randn(2, 2, 8, 8),
        ]
        target = torch.randint(0, 2, (2, 32, 32))
        loss = ds_loss(pred_list, target)
        results.append(("DeepSupervisionLoss", True, f"Loss: {loss.item():.4f}"))
    except Exception as e:
        results.append(("DeepSupervisionLoss", False, str(e)))
    
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            from src.io.stream_ops import write_mrc
            image_paths = []
            mask_paths = []
            for i in range(2):
                img = np.random.randn(256, 256).astype(np.float32)
                msk = ((np.random.rand(256, 256) > 0.95) * 255).astype(np.uint8)
                img_path = os.path.join(tmpdir, f'img_{i}.mrc')
                msk_path = os.path.join(tmpdir, f'msk_{i}.png')
                write_mrc(img_path, img)
                import cv2
                cv2.imwrite(msk_path, msk)
                image_paths.append(img_path)
                mask_paths.append(msk_path)
            
            dataset = CryoEMDataset(image_paths, mask_paths, patch_size=256, augment=True, use_patching=False)
            results.append(("CryoEMDataset create", True, f"Length: {len(dataset)}"))
            
            item = dataset[0]
            results.append(("CryoEMDataset __getitem__", True, 
                           f"Image: {item['image'].shape}, Mask: {item['mask'].shape}"))
    except Exception as e:
        results.append(("CryoEMDataset", False, f"{str(e)}\n{traceback.format_exc()}"))
    
    return results

def test_inference_module():
    results = []
    
    try:
        import torch
        from src.modeling.segmentation import LightResUNet_MSA
        from src.inference.export_onnx import export_to_onnx
        from src.inference.onnx_engine import ONNXInferenceEngine
        from src.inference.pipeline import InferencePipeline
        from src.preprocessing.pipeline import PreprocessingPipeline
        
        results.append(("Inference imports", True, ""))
    except Exception as e:
        results.append(("Inference imports", False, str(e)))
        return results
    
    try:
        model = LightResUNet_MSA(
            in_channels=1,
            out_channels=2,
            base_channels=16,
            num_res_blocks=2,
            attention_heads=2,
            attention_channels=32,
            use_deep_supervision=False
        )
        model.eval()
        results.append(("Create test model", True, f"Params: {sum(p.numel() for p in model.parameters()):,}"))
    except Exception as e:
        results.append(("Create test model", False, str(e)))
        return results
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_path = f.name
        
        try:
            input_shape = (1, 1, 128, 128)
            export_to_onnx(model, onnx_path, input_shape, simplify=True)
            results.append(("Export to ONNX", True, f"File: {os.path.basename(onnx_path)}, Size: {os.path.getsize(onnx_path)//1024}KB"))
        except Exception as e:
            os.unlink(onnx_path)
            results.append(("Export to ONNX", False, str(e)))
            return results
        
        try:
            engine = ONNXInferenceEngine(onnx_path, use_gpu=False)
            providers = engine.get_providers()
            input_shape = engine.get_input_shape()
            results.append(("ONNX Engine init", True, 
                           f"Providers: {providers[0]}, "
                           f"Input shape: {input_shape}"))
        except Exception as e:
            os.unlink(onnx_path)
            results.append(("ONNX Engine init", False, str(e)))
            return results
        
        try:
            test_input = np.random.randn(1, 1, 128, 128).astype(np.float32)
            import time as time_mod
            t0 = time_mod.perf_counter()
            result = engine.infer(test_input)
            infer_time = (time_mod.perf_counter() - t0) * 1000
            results.append(("ONNX Inference", True, 
                           f"Output shape: {result.shape}, "
                           f"Time: {infer_time:.2f}ms"))
        except Exception as e:
            results.append(("ONNX Inference", False, str(e)))
        
        try:
            bench_result = engine.benchmark((1, 1, 128, 128), num_runs=10)
            results.append(("ONNX Benchmark (10 runs)", True, 
                           f"Avg: {bench_result['mean_ms']:.2f}ms, "
                           f"Min: {bench_result['min_ms']:.2f}ms, "
                           f"Max: {bench_result['max_ms']:.2f}ms, "
                           f"FPS: {bench_result['throughput_fps']:.1f}"))
        except Exception as e:
            results.append(("ONNX Benchmark", False, str(e)))
        
        os.unlink(onnx_path)
        results.append(("Cleanup", True, ""))
        
    except Exception as e:
        results.append(("Inference test", False, f"Unexpected: {str(e)}\n{traceback.format_exc()}"))
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            tmp_onnx = f.name
        
        try:
            from src.modeling.segmentation import LightResUNet_MSA
            from src.inference.export_onnx import export_to_onnx
            small_model = LightResUNet_MSA(
                in_channels=1,
                out_channels=2,
                base_channels=16,
                num_res_blocks=2,
                attention_heads=2,
                attention_channels=32,
                use_deep_supervision=False
            )
            small_model.eval()
            export_to_onnx(small_model, tmp_onnx, (1, 1, 256, 256))
            
            preproc = PreprocessingPipeline(patch_size=128, overlap=16)
            pipeline = InferencePipeline(
                config_path="configs/config.yaml",
                onnx_model_path=tmp_onnx,
                preprocessing_pipeline=preproc
            )
            test_image = np.random.randn(256, 256).astype(np.float32)
            result = pipeline.process(test_image, use_patching=False)
            results.append(("InferencePipeline full", True, 
                           f"Particles: {result.num_particles}, "
                           f"Mask shape: {result.segmentation_mask.shape}"))
        finally:
            import time as t_sleep
            t_sleep.sleep(0.1)
            if os.path.exists(tmp_onnx):
                os.unlink(tmp_onnx)
    except Exception as e:
        results.append(("InferencePipeline", False, f"{str(e)}\n{traceback.format_exc()}"))
    
    return results

def test_postprocessing_module():
    results = []
    
    try:
        from src.postprocessing.detection import PeakDetector, NMS, extract_coordinates, detect_with_subpixel
        from src.postprocessing.refinement import ParticleRefiner, export_coordinates
        
        results.append(("Postprocessing imports", True, ""))
    except Exception as e:
        results.append(("Postprocessing imports", False, str(e)))
        return results
    
    try:
        prob_map = np.zeros((256, 256), dtype=np.float32)
        true_peaks = [(50, 50), (100, 120), (180, 200), (220, 50)]
        for x, y in true_peaks:
            prob_map[y, x] = 0.95
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    if 0 <= y+dy < 256 and 0 <= x+dx < 256:
                        dist = np.sqrt(dx*dx + dy*dy)
                        prob_map[y+dy, x+dx] = max(prob_map[y+dy, x+dx], 0.95 * np.exp(-dist*dist/8))
        
        prob_map += np.random.randn(256, 256) * 0.05
        prob_map = np.clip(prob_map, 0, 1)
        
        results.append(("Create test probability map", True, f"True peaks: {len(true_peaks)}"))
    except Exception as e:
        results.append(("Create test data", False, str(e)))
        return results
    
    try:
        detector = PeakDetector(threshold_abs=0.5, min_distance=20)
        peaks, scores = detector.detect(prob_map)
        results.append(("Peak Detection", True, f"Detected {len(peaks)} peaks"))
    except Exception as e:
        results.append(("Peak Detection", False, str(e)))
        peaks, scores = [], []
    
    try:
        nms = NMS(score_threshold=0.5)
        final_centers, final_scores = nms(peaks, scores, prob_map.shape)
        results.append(("NMS", True, f"After NMS: {len(final_centers)} peaks"))
    except Exception as e:
        results.append(("NMS", False, str(e)))
    
    try:
        mask = (prob_map > 0.5).astype(np.uint8)
        coords, scores_out, areas = extract_coordinates(mask)
        results.append(("Extract coordinates", True, f"Extracted {len(coords)} coordinates"))
    except Exception as e:
        results.append(("Extract coordinates", False, str(e)))
    
    try:
        coords_sub = detect_with_subpixel(prob_map, threshold=0.5, min_distance=20)
        results.append(("Sub-pixel detection", True, 
                       f"Detected {len(coords_sub)} particles, "
                       f"Avg score: {np.mean([p['score'] for p in coords_sub]):.3f}"))
    except Exception as e:
        results.append(("Sub-pixel detection", False, f"{str(e)}\n{traceback.format_exc()}"))
    
    try:
        refiner = ParticleRefiner()
        image = np.random.randn(256, 256).astype(np.float32)
        coords_list = [(int(p['x']), int(p['y'])) for p in coords_sub]
        scores_list = [p['score'] for p in coords_sub]
        refined = refiner.refine_coordinates(image, coords_list, scores_list)
        results.append(("Particle refinement (Gaussian fit)", True, 
                       f"Refined {len(refined)} particles"))
    except Exception as e:
        results.append(("Particle refinement", False, str(e)))
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            for fmt in ['star', 'csv', 'tsv', 'coords']:
                path = os.path.join(tmpdir, f'coords.{fmt}')
                export_coordinates(refined, path, fmt)
                if os.path.exists(path):
                    results.append((f"Export {fmt.upper()}", True, 
                                   f"File: {os.path.basename(path)}, Size: {os.path.getsize(path)} bytes"))
                else:
                    results.append((f"Export {fmt.upper()}", False, "File not created"))
    except Exception as e:
        results.append(("Export coordinates", False, f"{str(e)}\n{traceback.format_exc()}"))
    
    return results

def test_api_module():
    results = []
    
    try:
        from fastapi.testclient import TestClient
        from src.api.main import app
        from src.api.schemas import TaskStatus, PickingResult, Particle
        
        results.append(("API imports", True, ""))
    except Exception as e:
        results.append(("API imports", False, str(e)))
        return results
    
    try:
        client = TestClient(app)
        results.append(("FastAPI TestClient", True, ""))
    except Exception as e:
        results.append(("FastAPI TestClient", False, str(e)))
        return results
    
    try:
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        results.append(("GET /api/health", True, f"Status: {data.get('status')}"))
    except Exception as e:
        results.append(("GET /api/health", False, str(e)))
    
    try:
        response = client.get("/api/system/info")
        assert response.status_code == 200
        data = response.json()
        results.append(("GET /api/system/info", True, 
                       f"GPU available: {data.get('gpu_available')}, "
                       f"Model loaded: {data.get('model_loaded')}"))
    except Exception as e:
        results.append(("GET /api/system/info", False, str(e)))
    
    try:
        response = client.get("/api/pick/tasks?limit=10")
        assert response.status_code == 200
        data = response.json()
        results.append(("GET /api/pick/tasks", True, f"Tasks: {len(data)}"))
    except Exception as e:
        results.append(("GET /api/pick/tasks", False, str(e)))
    
    try:
        p = Particle(x=100.0, y=100.0, score=0.95, radius=30.0, snr=1.5)
        results.append(("Pydantic Particle", True, f"x={p.x}, y={p.y}, score={p.score}"))
    except Exception as e:
        results.append(("Pydantic models", False, str(e)))
    
    return results

def test_e2e_performance():
    results = []
    
    try:
        from src.preprocessing.pipeline import PreprocessingPipeline
        import torch
        from src.modeling.segmentation import LightResUNet_MSA
        from src.postprocessing.detection import detect_with_subpixel
        
        results.append(("E2E imports", True, ""))
    except Exception as e:
        results.append(("E2E imports", False, str(e)))
        return results
    
    try:
        sizes = [256, 512]
        pipeline = PreprocessingPipeline()
        
        model = LightResUNet_MSA(
            in_channels=1,
            out_channels=2,
            base_channels=16,
            num_res_blocks=2,
            attention_heads=2,
            attention_channels=32,
            use_deep_supervision=False
        )
        model.eval()
        
        for size in sizes:
            try:
                image = np.random.randn(size, size).astype(np.float32)
                
                t0 = time.time()
                p_result = pipeline.process(image)
                pre_time = (time.time() - t0) * 1000
                
                input_tensor = torch.from_numpy(p_result.image).unsqueeze(0).unsqueeze(0)
                
                t1 = time.time()
                with torch.no_grad():
                    output = model(input_tensor)
                infer_time = (time.time() - t1) * 1000
                
                if isinstance(output, tuple):
                    prob_map = torch.softmax(output[0], dim=1)[0, 1].numpy()
                else:
                    prob_map = torch.softmax(output, dim=1)[0, 1].numpy()
                
                t2 = time.time()
                coords = detect_with_subpixel(prob_map, threshold=0.5, min_distance=20)
                post_time = (time.time() - t2) * 1000
                
                total_time = pre_time + infer_time + post_time
                
                results.append((f"E2E Pipeline {size}x{size}", True,
                               f"Pre={pre_time:.1f}ms, "
                               f"Infer={infer_time:.1f}ms, "
                               f"Post={post_time:.1f}ms, "
                               f"Total={total_time:.1f}ms, "
                               f"Particles={len(coords)}"))
            except Exception as e:
                results.append((f"E2E Pipeline {size}x{size}", False, f"{str(e)}\n{traceback.format_exc()}"))
                continue
    except Exception as e:
        results.append(("E2E Performance", False, f"{str(e)}\n{traceback.format_exc()}"))
    
    return results

def main():
    print(f"\n{'=' * 60}")
    print("Cryo-EM Particle Picker - Integration Test Suite")
    print(f"{'=' * 60}")
    
    all_passed = 0
    all_failed = 0
    all_start = time.time()
    
    test_suites = [
        ("1. Imports & Dependencies", test_imports),
        ("2. IO Module (MRC Parser)", test_io_module),
        ("3. Preprocessing Module", test_preprocessing_module),
        ("4. Modeling Module", test_modeling_module),
        ("5. Training Module", test_training_module),
        ("6. Inference Module", test_inference_module),
        ("7. Postprocessing Module", test_postprocessing_module),
        ("8. API Module", test_api_module),
        ("9. E2E Performance", test_e2e_performance),
    ]
    
    for name, test_func in test_suites:
        p, f = run_test(name, test_func)
        all_passed += p
        all_failed += f
    
    total_elapsed = time.time() - all_start
    total = all_passed + all_failed
    
    print(f"\n{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
    print(f"\n{Colors.BOLD}FINAL RESULTS{Colors.ENDC}")
    print(f"  Total tests: {total}")
    print(f"  {Colors.GREEN}Passed:      {all_passed}{Colors.ENDC}")
    print(f"  {Colors.RED}Failed:      {all_failed}{Colors.ENDC}")
    print(f"  Success rate: {all_passed/total*100:.1f}%")
    print(f"  Total time: {total_elapsed:.2f}s")
    
    if all_failed == 0:
        print(f"\nAll tests passed!")
        return 0
    else:
        print(f"\nSome tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
