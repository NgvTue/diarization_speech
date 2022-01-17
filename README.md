# Giai đoạn dev 



# Tools 

## Tạo dữ liệu mixture dataset từ db

### Yêu cầu dữ liệu

1.  Dữ liệu nên đảm bảo trong 1 file utt độ lớn của silence không quá 1s.
2.  Dữ liệu phải có đầy đủ 4 file: segments, utt2spk, spk2utt, wav.scp 

### Chạy script 

chạy file python3 random_mixture.py -h xem các thông số 

Trong đó:
1. n_mixture: Số file mixture từ bộ dữ liệu mà mình muốn tạo ra.

2. n_speakers: Số speaker tối đa trong 1 file mixture được tạo. 

3. min_utts: Với mỗi speaker trong 1 file mixture được tạo thì cần ghép tối thiểu bơi bao nhiêu file utt.

4. max_utts: Với mỗi speaker trong 1 file mixture được tạo thì cần ghép tối đa bơi bao nhiêu file utt.

5. sil_scale: Độ dài trung bình khoảng cách ngắt nghỉ giữa 2 file utt trong cùng 1 speaker.

6. sil_scale_with_two_spk: Độ dài trung bình khoảng cách ngắt nghỉ giữa 2 file utt khác speaker.

7. overlap_scale: Độ dài trung bình overlap của toàn bộ dữ liệu mixture.

8. overlap_prob: Xác xuất xảy ra overlap giữa 2 utt khác speaker. (Khi ghép utt nếu 2 utt thuộc cùng 1 speaker thì sẽ tính khoảng cách sil vì cùng 1 speaker không thể nói cùng lúc. Nếu 2 utt khác speaker sẽ chia ra làm 2 TH là overlap hoặc không. Thì xác xuất xảy ra trường hợp overlap bẳng overlap_prob)

9. n_speakers_overlap: Nếu tồn tại đoạn overlap thì tối đa sẽ có bao nhiêu speaker nói trong đoạn đó.