table_template = """
|      Metric                   | Score (%) |
|:------------------------------|----------:|
| mIoU                          |   {mIoU:5.1f}   |
| mE_boundary_oU ↓              |   {mE_boundary_oU:5.1f}   |
| ├── mFP_boundary_oU ↓         |   {mFP_boundary_oU:5.1f}   |
| ├── mFN_boundary_oU ↓         |   {mFN_boundary_oU:5.1f}   |
| └── mE_boundary_oU_renormed ↓ |   {mE_boundary_oU_renormed:5.1f}   |
| mE_extent_oU ↓                |   {mE_extent_oU:5.1f}   |
| ├── mFP_extent_oU ↓           |   {mFP_extent_oU:5.1f}   |
| ├── mFN_extent_oU ↓           |   {mFN_extent_oU:5.1f}   |
| └── mE_extent_oU_renormed ↓   |   {mE_extent_oU_renormed:5.1f}   |
| mE_segment_oU ↓               |   {mE_segment_oU:5.1f}   |
| ├── mFP_segment_oU ↓          |   {mFP_segment_oU:5.1f}   |
| ├── mFN_segment_oU ↓          |   {mFN_segment_oU:5.1f}   |
| └── mE_segment_oU_renormed ↓  |   {mE_segment_oU_renormed:5.1f}   |
| mPrecision                    |   {mPrecision:5.1f}   |
| mRecall                       |   {mRecall:5.1f}   |
| mF1_score                     |   {mF1_score:5.1f}   |
| PixelAcc                      |   {PixelAcc:5.1f}   |
| mBoundaryIoU                  |   {mBoundaryIoU:5.1f}   |
"""
