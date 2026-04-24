Product specifications table for the NVIDIA GB200 NVL72, GB200 NVL4, and HGX B200.
Source: https://nvdam.widen.net/s/wwnsxrhm2w/blackwell-datasheet-3384703

### Product Specifications

| Feature | GB200 NVL72 | GB200 NVL4 | HGX B200 |
| :--- | :--- | :--- | :--- |
| **NVIDIA Blackwell GPUs \| Grace CPUs** | 72 \| 36 | 4 \| 2 | 8 \| 0 |
| **CPU Cores** | 2,592 Arm Neoverse V2 Cores | 144 Arm Neoverse V2 Cores | — |
| **Total NVFP4 Tensor Core** | 1,440 \| 720 PFLOPS | 80 \| 40 PFLOPS | 144 \| 72 PFLOPS |
| **Total FP8/FP6 Tensor Core** | 720 PFLOPS | 40 PFLOPS | 72 PFLOPS |
| **Total Fast Memory** | 31 TB | 1.8 TB | 1.4 TB |
| **Total Memory Bandwidth** | 576 TB/s | 32 TB/s | 62 TB/s |
| **Total NVLink Bandwidth** | 130 TB/s | 7.2 TB/s | 14.4 TB/s |

### Individual Blackwell GPU Specifications

| Feature | GB200 NVL72 (per GPU) | GB200 NVL4 (per GPU) | HGX B200 (per GPU) |
| :--- | :--- | :--- | :--- |
| **FP4 Tensor Core** | 20 PFLOPS | 20 PFLOPS | 18 PFLOPS |
| **FP8/FP6 Tensor Core** | 10 PFLOPS | 10 PFLOPS | 9 PFLOPS |
| **INT8 Tensor Core** | 10 POPS | 10 POPS | 9 POPS |
| **FP16/BF16 Tensor Core** | 5 PFLOPS | 5 PFLOPS | 4.5 PFLOPS |
| **TF32 Tensor Core** | 2.5 PFLOPS | 2.5 PFLOPS | 2.2 PFLOPS |
| **FP32** | 80 TFLOPS | 80 TFLOPS | 75 TFLOPS |
| **FP64 / FP64 Tensor Core** | 40 TFLOPS | 40 TFLOPS | 37 TFLOPS |
| **GPU Memory \| Bandwidth** | 186 GB HBM3E \| 8 TB/s | 186 GB HBM3E \| 8 TB/s | 180 GB HBM3E \| 7.7 TB/s |
| **Multi-Instance GPU (MIG)** | 7 | 7 | 7 |
| **Decompression Engine** | Yes | Yes | Yes |
| **Decoders** | 7 NVDEC, 7 nvJPEG | 7 NVDEC, 7 nvJPEG | 7 NVDEC, 7 nvJPEG |
| **Max Thermal Design Power (TDP)** | Configurable up to 1,200 W | Configurable up to 1,200 W | Configurable up to 1,000 W |
| **Interconnect** | 5th Gen NVLink: 1.8 TB/s; PCIe Gen5: 128 GB/s | 5th Gen NVLink: 1.8 TB/s; PCIe Gen5: 128 GB/s | 5th Gen NVLink: 1.8 TB/s; PCIe Gen5: 128 GB/s |
| **Server Options** | NVIDIA GB200 NVL72 partner and NVIDIA-Certified Systems with 72 GPUs | NVIDIA MGX partner and NVIDIA-Certified Systems | NVIDIA HGX B200 partner and NVIDIA-Certified Systems with 8 GPUs |

*Notes: Total NVFP4 Tensor Core specs shown as sparse | dense. All other Tensor Core specs are sparse; dense is one-half of the sparse value. Decoder speedups over NVIDIA H100 GPUs: 2x H.264, 1.25x HEVC, 1.25x VP9; AV1 support is new to NVIDIA Blackwell GPUs.*
