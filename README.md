# Goorm NLP Project 1 - Yelp data sentimental classification
Goorm 자연어 처리 전문가 양성과정 4기 첫 번째 프로젝트
1. 기간 : 2022.07.13~2022.07.19
2. 주제 : Yelp 식당 리뷰 데이터를 활용한 감성 분류 (Binary Classification)
3. 목표
    1) Baseline Code보다 Accuracy가 1%p 높은 모델 개발 (Baseline : 약 98%)
    2) NLP Process의 전반적인 이해
4. 성과
    1) Baseline Code보다 0.7%p 높은 모델 개발
    2) 최종 3위 (총 6개조)
    3) 6개조 중 baseline code bug 최초 해결
5. 환경 : Google Colab Pro+
6. 주요 라이브러리 : transformers, datasets, pandas, torch, re
7. 구성원 및 역할
    * 박정민 (팀장)
        * 팀 프로젝트 관리 감독 및 총괄
        * WandB 환경 구성 및 기타 Data Preprocessing 기법 제시
        * OverConfidence 문제 해결법 제시
    * 이예인
        * Pre-trained model 탐색 및 실험
        * Baseline code 정리 및 보충 분석
    * 이용호
        * 일정 조율, 확립 및 문서 정리 보조
        * Text classification 정보 검색 및 공유
    * 임영서
        * Batch size 및 Training data resampling을 통한 실험 시간 단축 방법 탐색
        * 실험 환경 기록, 정리
    * 정재빈
        * Skewed class 해결을 위한 Data Augmentation 제시
        * Hyperparameter turning
8. 핵심 아이디어 및 기술
    * WandB Sweep을 활용한 Hyperparameter 탐색 자동화
    * Label Smoothing을 통한 Over confidence 문제 해결
