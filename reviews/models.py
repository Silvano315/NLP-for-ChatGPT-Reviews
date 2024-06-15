from django.db import models

class Review(models.Model):
    username = models.CharField(max_length=100)
    review_text = models.TextField()
    predicted_score = models.IntegerField()
    satisfaction_response = models.CharField(max_length=50)

    def __str__(self):
        return f"{self.username} - {self.predicted_score}"
    
    class Meta:
        app_label = 'reviews'