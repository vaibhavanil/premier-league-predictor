document.addEventListener('DOMContentLoaded', function() {
    const addDispatchForm = document.getElementById('dispatchForm');
    const modal = document.getElementById('myModal');
    const editForm = document.getElementById('editForm');
    const dispatchTableBody = document.getElementById('dispatchTableBody');
    let dispatchArray = JSON.parse(localStorage.getItem('dispatchArray')) || [];

    function showModal(data) {
        modal.style.display = 'block';
        editForm.dataset.index = data.index; // Assign the index to the edit form dataset
        editForm.editDestination.value = data.destination;
        editForm.editJobNo.value = data.jobNo;
        editForm.editDcNo.value = data.dcNo;
        editForm.editInvoiceNo.value = data.invoiceNo;
        editForm.editClient.value = data.client;
        editForm.editCourier.value = data.courier;
        editForm.editTransactionNo.value = data.transactionNo;
        editForm.editNoOfBoxes.value = data.noOfBoxes;
    }

    function updateLocalStorage() {
        localStorage.setItem('dispatchArray', JSON.stringify(dispatchArray));
    }

    function renderTable() {
        dispatchTableBody.innerHTML = '';
        dispatchArray.forEach(function(data, index) {
            const row = dispatchTableBody.insertRow();
            row.innerHTML = `
                <td>${data.destination}</td>
                <td>${data.jobNo}</td>
                <td>${data.dcNo}</td>
                <td>${data.invoiceNo}</td>
                <td>${data.client}</td>
                <td>${data.courier}</td>
                <td>${data.transactionNo}</td>
                <td>${data.noOfBoxes}</td>
                <td>
                    <button class="editBtn">Edit</button>
                    <button class="deleteBtn">Delete</button>
                </td>
            `;
            row.dataset.index = index;
        });
    }

    addDispatchForm.addEventListener('submit', function(event) {
        event.preventDefault();
        const dispatchData = {
            destination: addDispatchForm.destination.value,
            jobNo: addDispatchForm.jobNo.value,
            dcNo: addDispatchForm.dcNo.value,
            invoiceNo: addDispatchForm.invoiceNo.value,
            client: addDispatchForm.client.value,
            courier: addDispatchForm.courier.value,
            transactionNo: addDispatchForm.transactionNo.value,
            noOfBoxes: addDispatchForm.noOfBoxes.value,
            index: dispatchArray.length // Assign the index to the dispatch data
        };
        dispatchArray.push(dispatchData);
        updateLocalStorage();
        showModal(dispatchData);
        renderTable();
        addDispatchForm.reset();
    });

    dispatchTableBody.addEventListener('click', function(event) {
        const target = event.target;
        if (target.classList.contains('editBtn')) {
            const index = target.closest('tr').dataset.index; // Retrieve the index from the dataset
            showModal(dispatchArray[index]);
        } else if (target.classList.contains('deleteBtn')) {
            const index = target.closest('tr').dataset.index; // Retrieve the index from the dataset
            dispatchArray.splice(index, 1);
            updateLocalStorage();
            renderTable();
        }
    });

    
    // Close modal when close button or outside modal is clicked
    modal.addEventListener('click', function(event) {
        if (event.target === modal || event.target.classList.contains('close')) {
            modal.style.display = 'none';
        }
    });

    renderTable();
});
