#!/bin/bash

# Script to reformat incomplete_sessions_ec2.txt from OAS2_XXXX_session_Y to OAS2_XXXX/session_Y format

INCOMPLETE_LIST="incomplete_sessions_ec2.txt"
TEMP_FILE="incomplete_sessions_temp.txt"

echo "🔄 Reformatting incomplete sessions list..."

# Check if the file exists
if [ ! -f "$INCOMPLETE_LIST" ]; then
    echo "❌ File not found: $INCOMPLETE_LIST"
    exit 1
fi

# Create backup
cp "$INCOMPLETE_LIST" "${INCOMPLETE_LIST}.backup"
echo "📋 Created backup: ${INCOMPLETE_LIST}.backup"

# Convert format: OAS2_XXXX_session_Y -> OAS2_XXXX/session_Y
sed 's/_session_/\//g' "$INCOMPLETE_LIST" > "$TEMP_FILE"

# Replace original with reformatted version
mv "$TEMP_FILE" "$INCOMPLETE_LIST"

echo "✅ Reformatting complete!"
echo "📊 Original format: OAS2_XXXX_session_Y"
echo "📊 New format: OAS2_XXXX/session_Y"

# Show first few lines as example
echo ""
echo "📄 First 5 lines of reformatted file:"
head -5 "$INCOMPLETE_LIST"

echo ""
echo "📄 Total lines: $(wc -l < "$INCOMPLETE_LIST")" 