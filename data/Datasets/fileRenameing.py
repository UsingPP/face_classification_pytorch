import os

def rename_images(base_dir=None):
    """
    폴더 구조: 폴더1/폴더1-1/이미지
    이미지를 폴더1-1-1.jpg, 폴더1-1-2.webp 형식으로 변경
    """
    if base_dir is None:
        # 스크립트가 위치한 폴더 (폴더1)를 기준으로 실행
        base_dir = os.path.dirname(os.path.abspath(__file__))
 
    # 이미지 확장자 목록
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tiff', '.svg'}
 
    renamed_count = 0
    skipped_count = 0
 
    for subfolder in sorted(os.listdir(base_dir)):
        subfolder_path = os.path.join(base_dir, subfolder)
 
        # 하위 폴더만 처리 (파일 제외)
        if not os.path.isdir(subfolder_path):
            continue
 
        # 해당 폴더 안의 이미지 파일 수집
        image_files = []
        for filename in os.listdir(subfolder_path):
            filepath = os.path.join(subfolder_path, filename)
            ext = os.path.splitext(filename)[1].lower()
            if os.path.isfile(filepath) and ext in IMAGE_EXTENSIONS:
                image_files.append(filename)
 
        # 파일명 정렬 (기존 이름 기준)
        image_files.sort()
 
        print(f"\n📁 {subfolder} ({len(image_files)}개 이미지)")
 
        for idx, filename in enumerate(image_files, start=1):
            ext = os.path.splitext(filename)[1].lower()
            new_name = f"{subfolder}{idx}{ext}"
            old_path = os.path.join(subfolder_path, filename)
            new_path = os.path.join(subfolder_path, new_name)
 
            if old_path == new_path:
                print(f"  ⏭️  {filename} (변경 불필요)")
                skipped_count += 1
                continue
 
            # 충돌 방지: 새 이름이 이미 존재하는 경우
            if os.path.exists(new_path):
                print(f"  ⚠️  건너뜀: {new_name} 이미 존재함 (원본: {filename})")
                skipped_count += 1
                continue
 
            os.rename(old_path, new_path)
            print(f"  ✅ {filename} → {new_name}")
            renamed_count += 1
 
    print(f"\n🎉 완료! 변경: {renamed_count}개 / 건너뜀: {skipped_count}개")
 
 
if __name__ == "__main__":
    rename_images()