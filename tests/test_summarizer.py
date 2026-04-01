import unittest

from nvbroadcast.ai.summarizer import MeetingSummarizer


class MeetingSummarizerTests(unittest.TestCase):
    def test_document_review_does_not_promote_form_text_to_action_items(self):
        text = """
        State of Florida Department of Highway Safety and Motor Vehicles Division of Motor Services.
        Submit this form to your local tax collector office.
        Notice of sale seller must complete section 1 and 3.
        Bill of sale seller and purchaser must complete section 1, 2, and 3.
        State law requires that you state the mileage in connection with the transfer of ownership.
        Failure to complete or providing a false statement may result in fines and imprisonment.
        """
        notes = MeetingSummarizer().summarize(text, duration_seconds=120)
        self.assertIn("document review", notes.summary)
        self.assertEqual(notes.action_items, [])
        self.assertTrue(notes.key_points)
        self.assertEqual(notes.title, "Notice of sale and bill of sale form review")

    def test_repetitive_noise_is_filtered_out(self):
        text = """
        Thank you. Thank you. Thank you.
        If you have a question if you have a question if you have a question if you have a question.
        We need to send the revised draft by Friday.
        Let's review the launch checklist tomorrow.
        """
        notes = MeetingSummarizer().summarize(text, duration_seconds=300)
        joined_points = " ".join(notes.key_points).lower()
        self.assertNotIn("if you have a question if you have a question", joined_points)
        self.assertEqual(len(notes.action_items), 2)
        self.assertTrue(any("friday" in item.lower() for item in notes.action_items))

    def test_terms_review_gets_topic_title(self):
        text = """
        PayPal terms and conditions apply to the balance account.
        Apple Pay is a service provided by Apple Payment Services LLC.
        The card is linked to your PayPal balance account.
        Google Pay is a trademark of Google LLC.
        Rewards can be redeemed for cash or other redemption options.
        """
        notes = MeetingSummarizer().summarize(text, duration_seconds=180)
        self.assertIn("terms review", notes.summary)
        self.assertEqual(notes.title, "PayPal and Apple Pay terms overview")
        self.assertEqual(notes.action_items, [])


if __name__ == "__main__":
    unittest.main()
