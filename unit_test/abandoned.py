# class TestValidation(TestCase):
#     def test_cross_entropy(self):
#         for p in [0.1 * i for i in range(1, 10 + 1)]:
#             phi_when_bad = torch.tensor([1 - p, p])
#             phi_when_good = torch.tensor([0.0, 1.0])
#             cross_entropy = calculate_cross_entropy(phi_when_good, phi_when_bad)
#             # self.assertIs(cross_entropy, torch.tensor())
#             print('\n')
#             print(p, cross_entropy)

