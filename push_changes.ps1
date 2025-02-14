Set-Location -Path $PSScriptRoot
git add "scripts/visualize_analysis.py"
git add "requirements.txt"
git add "docs/"
git commit -m "Enhanced dashboards and documentation

1. Updated visualize_analysis.py:
   - Fixed both main and recommendations dashboards
   - Added proper threading support
   - Improved error handling
   - Enhanced data visualization

2. Updated requirements.txt:
   - Updated package versions
   - Added missing dependencies
   - Ensured compatibility

3. Added documentation:
   - Added comprehensive dashboard documentation
   - Updated progress reports
   - Added usage instructions"
git push
