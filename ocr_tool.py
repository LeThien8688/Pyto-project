#!/usr/bin/env python3
"""
OCR Tool - Công cụ nhận dạng văn bản sử dụng Vision framework cho iOS/Pyto

Sử dụng Apple Vision framework để thực hiện OCR (Optical Character Recognition)
trên các hình ảnh. Hỗ trợ nhiều ngôn ngữ và các chế độ nhận dạng khác nhau.

Yêu cầu:
- Pyto app trên iOS
- iOS 13.0 trở lên (Vision framework với VNRecognizeTextRequest)

Cách sử dụng:
    from ocr_tool import OCRTool

    ocr = OCRTool()
    text = ocr.recognize_text("path/to/image.jpg")
    print(text)
"""

import os
from typing import Optional, List, Tuple
from enum import Enum


class RecognitionLevel(Enum):
    """Mức độ nhận dạng văn bản"""
    FAST = "fast"       # Nhanh nhưng độ chính xác thấp hơn
    ACCURATE = "accurate"  # Chậm hơn nhưng chính xác hơn


class OCRTool:
    """
    Công cụ OCR sử dụng Vision framework của Apple.

    Attributes:
        recognition_level: Mức độ nhận dạng (fast hoặc accurate)
        languages: Danh sách ngôn ngữ để nhận dạng
        min_confidence: Độ tin cậy tối thiểu (0.0 - 1.0)

    Example:
        >>> ocr = OCRTool(recognition_level=RecognitionLevel.ACCURATE)
        >>> text = ocr.recognize_text("photo.jpg")
        >>> print(text)
    """

    def __init__(
        self,
        recognition_level: RecognitionLevel = RecognitionLevel.ACCURATE,
        languages: Optional[List[str]] = None,
        min_confidence: float = 0.0
    ):
        """
        Khởi tạo OCR Tool.

        Args:
            recognition_level: Mức độ nhận dạng (FAST hoặc ACCURATE)
            languages: Danh sách mã ngôn ngữ (ví dụ: ["vi-VN", "en-US"])
                      Nếu None, sẽ tự động phát hiện ngôn ngữ
            min_confidence: Độ tin cậy tối thiểu để chấp nhận kết quả (0.0 - 1.0)
        """
        self.recognition_level = recognition_level
        self.languages = languages
        self.min_confidence = min_confidence

        # Lazy loading các module iOS
        self._objc = None
        self._Vision = None
        self._UIKit = None
        self._Foundation = None
        self._is_ios = None

    @property
    def is_ios(self) -> bool:
        """Kiểm tra xem đang chạy trên iOS/Pyto không"""
        if self._is_ios is None:
            try:
                import objc
                self._is_ios = True
            except ImportError:
                try:
                    from rubicon.objc import ObjCClass
                    self._is_ios = True
                except ImportError:
                    self._is_ios = False
        return self._is_ios

    def _load_ios_modules(self):
        """Load các module iOS cần thiết"""
        if self._objc is not None:
            return

        try:
            # Thử import objc module của Pyto
            import objc
            self._objc = objc

            # Load Vision framework
            from objc import load_framework
            load_framework("Vision")

            # Import các class cần thiết
            self._VNRecognizeTextRequest = objc.objc_class("VNRecognizeTextRequest")
            self._VNImageRequestHandler = objc.objc_class("VNImageRequestHandler")
            self._UIImage = objc.objc_class("UIImage")
            self._NSURL = objc.objc_class("NSURL")
            self._NSData = objc.objc_class("NSData")

        except ImportError:
            try:
                # Thử rubicon-objc (backup option)
                from rubicon.objc import ObjCClass, load_library

                load_library("Vision")

                self._VNRecognizeTextRequest = ObjCClass("VNRecognizeTextRequest")
                self._VNImageRequestHandler = ObjCClass("VNImageRequestHandler")
                self._UIImage = ObjCClass("UIImage")
                self._NSURL = ObjCClass("NSURL")
                self._NSData = ObjCClass("NSData")

            except ImportError:
                raise RuntimeError(
                    "Không thể import objc module. "
                    "Tool này yêu cầu chạy trên iOS với Pyto app."
                )

    def _get_recognition_level_constant(self):
        """Lấy constant cho recognition level"""
        # VNRequestTextRecognitionLevel
        # 0 = accurate, 1 = fast
        if self.recognition_level == RecognitionLevel.FAST:
            return 1
        return 0

    def _load_image(self, image_path: str):
        """
        Load ảnh từ đường dẫn file.

        Args:
            image_path: Đường dẫn đến file ảnh

        Returns:
            UIImage object hoặc None nếu không load được
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Không tìm thấy file: {image_path}")

        # Chuyển sang đường dẫn tuyệt đối
        abs_path = os.path.abspath(image_path)

        try:
            # Sử dụng NSData và UIImage
            data = self._NSData.dataWithContentsOfFile_(abs_path)
            if data is None:
                raise ValueError(f"Không thể đọc file: {image_path}")

            image = self._UIImage.imageWithData_(data)
            if image is None:
                raise ValueError(f"Không thể parse ảnh: {image_path}")

            return image
        except Exception as e:
            raise ValueError(f"Lỗi khi load ảnh: {e}")

    def _create_request_handler(self, image):
        """
        Tạo VNImageRequestHandler từ UIImage.

        Args:
            image: UIImage object

        Returns:
            VNImageRequestHandler object
        """
        # Lấy CGImage từ UIImage
        cg_image = image.CGImage
        if cg_image is None:
            raise ValueError("Không thể lấy CGImage từ UIImage")

        # Tạo handler với options
        options = {}
        handler = self._VNImageRequestHandler.alloc().initWithCGImage_options_(
            cg_image, options
        )

        return handler

    def recognize_text(
        self,
        image_path: str,
        custom_words: Optional[List[str]] = None
    ) -> str:
        """
        Nhận dạng văn bản từ ảnh.

        Args:
            image_path: Đường dẫn đến file ảnh
            custom_words: Danh sách từ tùy chỉnh để cải thiện nhận dạng

        Returns:
            Văn bản được nhận dạng từ ảnh

        Example:
            >>> ocr = OCRTool()
            >>> text = ocr.recognize_text("document.jpg")
            >>> print(text)
        """
        if not self.is_ios:
            return self._recognize_text_fallback(image_path)

        self._load_ios_modules()

        # Load ảnh
        image = self._load_image(image_path)

        # Tạo request handler
        handler = self._create_request_handler(image)

        # Tạo text recognition request
        request = self._VNRecognizeTextRequest.alloc().init()

        # Cấu hình request
        request.setRecognitionLevel_(self._get_recognition_level_constant())

        # Thiết lập ngôn ngữ nếu có
        if self.languages:
            request.setRecognitionLanguages_(self.languages)

        # Thiết lập custom words nếu có
        if custom_words:
            request.setCustomWords_(custom_words)

        # Thực hiện request
        error = None
        success = handler.performRequests_error_([request], error)

        if not success:
            raise RuntimeError(f"OCR request thất bại: {error}")

        # Lấy kết quả
        results = request.results
        if not results:
            return ""

        # Xử lý kết quả
        recognized_texts = []
        for observation in results:
            # Lấy top candidate
            candidates = observation.topCandidates_(1)
            if candidates and len(candidates) > 0:
                candidate = candidates[0]
                confidence = candidate.confidence

                # Kiểm tra độ tin cậy
                if confidence >= self.min_confidence:
                    text = str(candidate.string)
                    recognized_texts.append(text)

        return "\n".join(recognized_texts)

    def recognize_text_with_locations(
        self,
        image_path: str
    ) -> List[Tuple[str, float, Tuple[float, float, float, float]]]:
        """
        Nhận dạng văn bản với vị trí bounding box.

        Args:
            image_path: Đường dẫn đến file ảnh

        Returns:
            List các tuple (text, confidence, (x, y, width, height))
            Tọa độ được chuẩn hóa từ 0.0 đến 1.0

        Example:
            >>> ocr = OCRTool()
            >>> results = ocr.recognize_text_with_locations("document.jpg")
            >>> for text, conf, bbox in results:
            ...     print(f"{text} (confidence: {conf:.2f})")
        """
        if not self.is_ios:
            # Fallback không hỗ trợ locations
            text = self._recognize_text_fallback(image_path)
            if text:
                return [(text, 1.0, (0.0, 0.0, 1.0, 1.0))]
            return []

        self._load_ios_modules()

        # Load ảnh
        image = self._load_image(image_path)

        # Tạo request handler
        handler = self._create_request_handler(image)

        # Tạo text recognition request
        request = self._VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(self._get_recognition_level_constant())

        if self.languages:
            request.setRecognitionLanguages_(self.languages)

        # Thực hiện request
        error = None
        success = handler.performRequests_error_([request], error)

        if not success:
            raise RuntimeError(f"OCR request thất bại: {error}")

        # Lấy kết quả với locations
        results = []
        observations = request.results

        if observations:
            for observation in observations:
                candidates = observation.topCandidates_(1)
                if candidates and len(candidates) > 0:
                    candidate = candidates[0]
                    confidence = float(candidate.confidence)

                    if confidence >= self.min_confidence:
                        text = str(candidate.string)

                        # Lấy bounding box
                        bbox = observation.boundingBox
                        x = float(bbox.origin.x)
                        y = float(bbox.origin.y)
                        width = float(bbox.size.width)
                        height = float(bbox.size.height)

                        results.append((text, confidence, (x, y, width, height)))

        return results

    def get_supported_languages(self) -> List[str]:
        """
        Lấy danh sách ngôn ngữ được hỗ trợ.

        Returns:
            List các mã ngôn ngữ được hỗ trợ

        Example:
            >>> ocr = OCRTool()
            >>> languages = ocr.get_supported_languages()
            >>> print(languages)
            ['en-US', 'vi-VN', 'zh-Hans', ...]
        """
        if not self.is_ios:
            # Trả về các ngôn ngữ phổ biến khi không ở iOS
            return [
                "en-US", "vi-VN", "zh-Hans", "zh-Hant",
                "ja-JP", "ko-KR", "fr-FR", "de-DE",
                "es-ES", "pt-BR", "it-IT", "ru-RU"
            ]

        self._load_ios_modules()

        try:
            # Tạo request để lấy supported languages
            request = self._VNRecognizeTextRequest.alloc().init()
            request.setRecognitionLevel_(self._get_recognition_level_constant())

            # iOS 14+ có supportedRecognitionLanguages
            error = None
            languages = request.supportedRecognitionLanguagesAndReturnError_(error)

            if languages:
                return [str(lang) for lang in languages]
        except Exception:
            pass

        # Fallback cho iOS cũ hơn
        return ["en-US", "vi-VN"]

    def _recognize_text_fallback(self, image_path: str) -> str:
        """
        Fallback OCR khi không chạy trên iOS.
        Sử dụng pytesseract hoặc easyocr nếu có.

        Args:
            image_path: Đường dẫn đến file ảnh

        Returns:
            Văn bản được nhận dạng
        """
        # Thử pytesseract
        try:
            import pytesseract
            from PIL import Image

            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, lang='vie+eng')
            return text.strip()
        except ImportError:
            pass

        # Thử easyocr
        try:
            import easyocr

            reader = easyocr.Reader(['vi', 'en'])
            results = reader.readtext(image_path)
            texts = [result[1] for result in results]
            return "\n".join(texts)
        except ImportError:
            pass

        raise RuntimeError(
            "Không thể thực hiện OCR. "
            "Vui lòng chạy trên iOS với Pyto hoặc cài đặt pytesseract/easyocr."
        )

    def recognize_from_photos(self) -> Optional[str]:
        """
        Mở Photos picker và nhận dạng văn bản từ ảnh được chọn.
        Chỉ hoạt động trên iOS/Pyto.

        Returns:
            Văn bản được nhận dạng hoặc None nếu không chọn ảnh

        Example:
            >>> ocr = OCRTool()
            >>> text = ocr.recognize_from_photos()
            >>> if text:
            ...     print(text)
        """
        if not self.is_ios:
            raise RuntimeError("Chức năng này chỉ hoạt động trên iOS/Pyto")

        try:
            import photos
            import tempfile
            import uuid

            # Mở photo picker
            picked = photos.pick_image()
            if picked is None:
                return None

            # Lưu tạm thời
            temp_path = os.path.join(
                tempfile.gettempdir(),
                f"ocr_temp_{uuid.uuid4().hex}.jpg"
            )

            try:
                picked.save(temp_path, "JPEG")
                result = self.recognize_text(temp_path)
                return result
            finally:
                # Xóa file tạm
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        except ImportError:
            raise RuntimeError("Module 'photos' không khả dụng")

    def recognize_from_camera(self) -> Optional[str]:
        """
        Mở camera và nhận dạng văn bản từ ảnh chụp.
        Chỉ hoạt động trên iOS/Pyto.

        Returns:
            Văn bản được nhận dạng hoặc None nếu không chụp ảnh

        Example:
            >>> ocr = OCRTool()
            >>> text = ocr.recognize_from_camera()
            >>> if text:
            ...     print(text)
        """
        if not self.is_ios:
            raise RuntimeError("Chức năng này chỉ hoạt động trên iOS/Pyto")

        try:
            import camera
            import tempfile
            import uuid

            # Chụp ảnh
            captured = camera.take_photo()
            if captured is None:
                return None

            # Lưu tạm thời
            temp_path = os.path.join(
                tempfile.gettempdir(),
                f"ocr_temp_{uuid.uuid4().hex}.jpg"
            )

            try:
                captured.save(temp_path, "JPEG")
                result = self.recognize_text(temp_path)
                return result
            finally:
                # Xóa file tạm
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        except ImportError:
            raise RuntimeError("Module 'camera' không khả dụng")


def quick_ocr(image_path: str, accurate: bool = True) -> str:
    """
    Hàm tiện ích để OCR nhanh.

    Args:
        image_path: Đường dẫn đến file ảnh
        accurate: True để dùng chế độ chính xác, False để nhanh

    Returns:
        Văn bản được nhận dạng

    Example:
        >>> from ocr_tool import quick_ocr
        >>> text = quick_ocr("document.jpg")
        >>> print(text)
    """
    level = RecognitionLevel.ACCURATE if accurate else RecognitionLevel.FAST
    ocr = OCRTool(recognition_level=level)
    return ocr.recognize_text(image_path)


# Demo khi chạy trực tiếp
if __name__ == "__main__":
    import sys

    print("=" * 50)
    print("OCR Tool - Vision Framework")
    print("=" * 50)

    # Kiểm tra môi trường
    ocr = OCRTool()
    print(f"\nĐang chạy trên iOS: {ocr.is_ios}")

    # Lấy ngôn ngữ hỗ trợ
    print("\nNgôn ngữ hỗ trợ:")
    for lang in ocr.get_supported_languages()[:5]:
        print(f"  - {lang}")
    print("  ...")

    # Nếu có argument là đường dẫn ảnh
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"\nĐang OCR file: {image_path}")
        print("-" * 50)

        try:
            text = ocr.recognize_text(image_path)
            print(text if text else "(Không tìm thấy văn bản)")
        except Exception as e:
            print(f"Lỗi: {e}")
    else:
        print("\nCách sử dụng:")
        print("  python ocr_tool.py <đường_dẫn_ảnh>")
        print("\nVí dụ:")
        print("  python ocr_tool.py document.jpg")
        print("\nHoặc import trong code:")
        print("  from ocr_tool import OCRTool, quick_ocr")
        print("  text = quick_ocr('image.jpg')")
