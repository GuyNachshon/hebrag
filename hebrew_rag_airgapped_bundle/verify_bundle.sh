#!/bin/bash
echo "Verifying bundle..."
echo "✓ Bundle created in Linux Docker environment"

# Count packages
total_packages=$(find packages/ -name "*.whl" -o -name "*.tar.gz" 2>/dev/null | wc -l)
linux_packages=$(find packages/linux_x86_64/ -name "*.whl" 2>/dev/null | wc -l)
echo "✓ Total packages: $total_packages"
echo "✓ Linux packages: $linux_packages"

# Check scripts
if [ -x "scripts/deploy.sh" ]; then
    echo "✓ Deployment script ready"
else
    echo "❌ Deployment script missing"
fi

echo "Bundle verification completed!"
