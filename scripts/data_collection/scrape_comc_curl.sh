#!/bin/bash

# COMC PSA Card Image Scraper using curl
# Downloads PSA graded baseball card images

OUTPUT_DIR="data/training"
IMAGES_PER_GRADE=30
USER_AGENT="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# Grades to scrape
declare -A GRADES
GRADES["PSA_10"]="PSA+10"
GRADES["PSA_9"]="PSA+9"
GRADES["PSA_8"]="PSA+8"
GRADES["PSA_7"]="PSA+7"
GRADES["PSA_6"]="PSA+6"
GRADES["PSA_5"]="PSA+5"
GRADES["PSA_4"]="PSA+4"
GRADES["PSA_3"]="PSA+3"
GRADES["PSA_2"]="PSA+2"
GRADES["PSA_1"]="PSA+1"

echo "========================================"
echo "COMC PSA Card Image Scraper"
echo "========================================"

for folder in "${!GRADES[@]}"; do
    grade="${GRADES[$folder]}"
    grade_dir="$OUTPUT_DIR/$folder"
    
    echo ""
    echo "========================================"
    echo "Scraping $grade -> $folder"
    echo "========================================"
    
    # Count existing images
    existing=$(ls -1 "$grade_dir"/*.{jpg,png,jpeg,PNG,JPG,JPEG} 2>/dev/null | wc -l)
    echo "Existing images: $existing"
    
    if [ "$existing" -ge "$IMAGES_PER_GRADE" ]; then
        echo "Already have enough images, skipping"
        continue
    fi
    
    needed=$((IMAGES_PER_GRADE - existing))
    echo "Need: $needed more images"
    
    downloaded=0
    
    for page in 1 2 3 4 5; do
        if [ "$downloaded" -ge "$needed" ]; then
            break
        fi
        
        url="https://www.comc.com/Cards/Baseball,sp=$grade,pg=$page"
        echo "Fetching page $page..."
        
        # Get page content and extract image URLs
        urls=$(curl -s -L "$url" -A "$USER_AGENT" 2>/dev/null | grep -oE 'https://img\.comc\.com/i/[^"]+\.(jpg|png|jpeg)' | sort -u | head -50)
        
        if [ -z "$urls" ]; then
            echo "No images found on page $page"
            continue
        fi
        
        url_count=$(echo "$urls" | wc -l)
        echo "Found $url_count image URLs"
        
        # Download images
        for img_url in $urls; do
            if [ "$downloaded" -ge "$needed" ]; then
                break
            fi
            
            img_num=$((existing + downloaded + 1))
            filename=$(printf "comc_%03d.jpg" $img_num)
            save_path="$grade_dir/$filename"
            
            if [ -f "$save_path" ]; then
                continue
            fi
            
            echo -n "  Downloading $filename... "
            if curl -s -L "$img_url" -A "$USER_AGENT" -o "$save_path" 2>/dev/null; then
                # Check if file is valid (at least 1KB)
                size=$(stat -f%z "$save_path" 2>/dev/null || stat -c%s "$save_path" 2>/dev/null)
                if [ "$size" -gt 1000 ]; then
                    downloaded=$((downloaded + 1))
                    echo "OK ($downloaded/$needed)"
                else
                    rm -f "$save_path"
                    echo "FAILED (too small)"
                fi
            else
                echo "FAILED"
            fi
            
            sleep 0.3
        done
        
        sleep 1
    done
    
    final_count=$(ls -1 "$grade_dir"/*.{jpg,png,jpeg,PNG,JPG,JPEG} 2>/dev/null | wc -l)
    echo "Total images for $folder: $final_count"
done

echo ""
echo "========================================"
echo "SCRAPING COMPLETE"
echo "========================================"

# Summary
echo ""
echo "Summary:"
for folder in PSA_10 PSA_9 PSA_8 PSA_7 PSA_6 PSA_5 PSA_4 PSA_3 PSA_2 PSA_1; do
    count=$(ls -1 "$OUTPUT_DIR/$folder"/*.{jpg,png,jpeg,PNG,JPG,JPEG} 2>/dev/null | wc -l)
    echo "  $folder: $count images"
done
